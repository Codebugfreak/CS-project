from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from models import db, User, Plant, SystemStatus,PhytochemicalCache, UserFormulation 
from flask_sqlalchemy import SQLAlchemy
from bcrypt import hashpw, gensalt, checkpw
from datetime import datetime, timezone
import os
from project import happy_birthday, normalize_path
from deep_dream import generate_visual
from deep_dream import LAYER_METADATA
import logging
import pandas as pd
from phyto import fetch_phytochemical_info, classify_phytochemical, parse_smiles,get_benchmark_smiles,CONFIG
import re
from scent import (
    fetch_ingredient_details,
    calculate_compatibility,
    select_phytochemicals,
    suggest_formulation,
    dataset,
    get_mean_molecular_weight,
    fallback_atom_count_score
)
import tensorflow as tf
from train_initial_model import pubchem_df,MODEL_SAVE_PATH,EXCLUDED_PHENOLS,load_trained_model
import uuid
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLAlchemy configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'users.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with app
db.init_app(app)


# Global variable to hold the model
model = load_trained_model()


# Function to read CSV and insert data into database only once
def import_plants_from_excel(excel_path):
    with app.app_context():
        status = SystemStatus.query.first()
        if status and status.csv_imported:
            print("Excel data already imported. Skipping import.")
            return
        
        try:
            # Read Excel file into DataFrame
            plant_df = pd.read_excel(excel_path, engine="openpyxl")  # Read .xlsx correctly
            
            # Debugging: Print detected columns
            print(f"Excel Columns Detected: {plant_df.columns.tolist()}")

            # Rename columns if necessary
            expected_columns = ["Plant Name", "Scientific Name", "Common Uses", "Phytochemicals"]
            if list(plant_df.columns) != expected_columns:
                print("⚠️ Excel headers are incorrect. Attempting to rename.")
                plant_df.columns = expected_columns

            for _, row in plant_df.iterrows():
                plant_name = row["Plant Name"].strip()
                scientific_name = row["Scientific Name"].strip()
                common_uses = row["Common Uses"].strip()
                phytochemicals = row["Phytochemicals"].strip()

                if not all([plant_name, scientific_name, phytochemicals]):
                    print(f"⚠️ Skipping invalid row: {row}")
                    continue

                existing_plant = Plant.query.filter_by(plant_name=plant_name).first()
                if not existing_plant:
                    new_plant = Plant(
                        plant_name=plant_name,
                        scientific_name=scientific_name,
                        common_uses=common_uses,
                        phytochemicals=phytochemicals
                    )
                    db.session.add(new_plant)

            if not status:
                status = SystemStatus(csv_imported=True)
                db.session.add(status)
            else:
                status.csv_imported = True

            db.session.commit()
            print("Excel data successfully imported into users.db.")

        except Exception as e:
            print(f"⚠️ Error importing Excel data: {e}")

# Create the database tables
with app.app_context():
    db.create_all()



# Home route
@app.route("/")
def home():
    birthday_message = None
    if "user_name" in session:
        current_date = datetime.now().strftime("%m-%d")
        user = User.query.filter_by(name=session["user_name"]).first()
        if user and user.birthday == current_date:
            birthday_message = happy_birthday(user.name)
    return render_template("index.html", birthday=birthday_message)

# Profile route (modify with cs50 assignment)
@app.route('/profile')
def profile():
    # Ensure the user is logged in
    if 'user_name' not in session:
        flash("You must be logged in to view your profile.", "danger")
        return redirect(url_for('login'))

    # Fetch the logged-in user
    user = User.query.filter_by(name=session['user_name']).first()
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('logout'))
    
    # Fetch the user's formulations
    formulations = UserFormulation.query.filter_by(user_id=user.id).order_by(UserFormulation.created_at.desc()).all()

    # Normalize the profile picture path for rendering
    if user.profile_pic:
        profile_pic_url = normalize_path(url_for('static', filename=user.profile_pic))  # Uses correct relative path
    else:
        profile_pic_url = "https://via.placeholder.com/150"  # Default profile pic

    session.pop('_flashes', None)  # Clears flash messages before redirecting
    
    # Render the profile template
    return render_template(
        'profile.html',
        user_name=user.name,
        email=user.email,
        birthday=user.birthday,
        bio=user.bio,
        profile_pic=profile_pic_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        formulations=formulations
    )


@app.route('/deep_dream', methods=['GET', 'POST'])
def deep_dream():
    metadata = LAYER_METADATA
    if request.method == 'POST':
        image = request.files['image']
        if image:
            # Save the uploaded image
            upload_dir = os.path.join(BASE_DIR, 'static/uploads')
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, image.filename)
            image.save(image_path)

            # Get user-specified parameters
            model_name = request.form.get("model_name", None)
            layer_name = request.form.get("layer_name", None)

            print(f"Received model_name: {model_name}, layer_name: {layer_name}")

            gradient_multiplier = float(request.form.get("gradient_multiplier", 0.01))
            iterations = int(request.form.get("iterations", 1))
            apply_filter = request.form.get("apply_filter", None)
        
            # Generate the deep dream image
            output_path = generate_visual(
                image_path,
                model_name=model_name,
                layer_name=layer_name,
                gradient_multiplier=gradient_multiplier,
                iterations=iterations,
                apply_filter=apply_filter,
            )


            if output_path:
                # Normalize the path for URL usage
                rel_path = output_path.replace("\\", "/").split("static/")[-1]
                return render_template('deep_dream.html', output_path=rel_path, metadata=metadata)
            else:
                # Handle errors gracefully
                flash("Error generating the deep dream image.", "danger")
                return redirect(url_for('deep_dream'))

    # Render the form initially with metadata for the dropdown options
    return render_template('deep_dream.html', metadata=metadata)


# delete deep_dream image
@app.route('/delete_image', methods=['POST'])
def delete_image():
    image_path = request.form.get('image_path')  # Get the image path from the form
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)  # Delete the image file
            flash("Image deleted successfully.", "success")
        except Exception as e:
            flash(f"Error deleting image: {e}", "danger")
    else:
        flash("Image not found or invalid path.", "danger")
    session.pop('_flashes', None)  # Clears flash messages before redirecting
    return redirect(url_for('deep_dream'))



# Standard email validation regex pattern
EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_name = request.form['user_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        birthday = request.form['birthday']
        bio = request.form.get('bio', '')

        # Validate passwords
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        # Validate email format
        if not re.match(EMAIL_REGEX, email):
            flash("Invalid email address. Please enter a valid email.", "danger")
            return render_template('register.html')

        # Check if username or email already exists
        if User.query.filter_by(name=user_name).first():
            flash("Username already exists.", "danger")
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template('register.html')

        # Hash the password
        hashed_password = hashpw(password.encode('utf-8'), gensalt())

        # Handle profile picture upload
        profile_pic = request.files.get('profile_pic')
        profile_pic_path = None
        if profile_pic:
            upload_dir = os.path.join(BASE_DIR, 'static/uploads')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure directory exists
            
            # Generate a unique filename
            file_extension = os.path.splitext(profile_pic.filename)[1]  # Get file extension
            unique_filename = f"{uuid.uuid4().hex}{file_extension}"  # Generate unique name
            profile_pic_path = f"uploads/{unique_filename}"  # Store relative path

            # Save the image in static/uploads/
            profile_pic.save(os.path.join(upload_dir, unique_filename))

        # Add user to the database
        new_user = User(
            name=user_name,
            email=email,
            password=hashed_password.decode('utf-8'),  # Store as string in the database
            birthday=birthday,
            bio=bio,
            profile_pic=profile_pic_path  # Store the relative path (e.g., uploads/unique.jpg)
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful.", "success")
        session.pop('_flashes', None)  # Clears flash messages before redirecting
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_name = request.form['user_name']
        password = request.form['password']

        # Validate user credentials
        user = User.query.filter_by(name=user_name).first()
        if user and checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            session['user_name'] = user_name
            flash("Login successful.", "success")
            # Import plants from EXCEL only once after login
            excel_path = normalize_path(os.path.join(BASE_DIR, "data", "cosmetic_plants_complete.xlsx"))
            import_plants_from_excel(excel_path)
            session.pop('_flashes', None)  # Clears flash messages before redirecting
            return redirect(url_for('home'))

        flash("Invalid username or password.", "danger")
        return render_template("login.html")  # Allows flash message to show before redirect
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_name', None)
    flash("You have been logged out.", "info")
    session.pop('_flashes', None)  # Clears flash messages before redirecting
    return redirect(url_for('home'))


# Initialize logging
logging.basicConfig(level=logging.DEBUG)


@app.route("/phytochemical", methods=["GET", "POST"])
def phytochemical():
    if request.method == "POST":
        query = request.form.get("query")
        if not query:
            flash("Please provide a phytochemical name or SMILES string.", "danger")
            return render_template("phyto.html")

        # Fetch compound data
        compound_info = fetch_phytochemical_info(query)
        if not compound_info:
            flash("Compound not found in PubChem.", "danger")
            return render_template("phyto.html")

        # Classify compound
        classification = classify_phytochemical(compound_info["canonical_smiles"])

        # Get benchmark SMILES for the class
        benchmark_smiles = get_benchmark_smiles(classification)

        # Parse benchmark SMILES
        weights = CONFIG  # Use the CONFIG dictionary as weights
        benchmark_data = parse_smiles(benchmark_smiles, weights=weights)

        if not benchmark_data or "compound_score" not in benchmark_data:
            benchmark_score = None
            print(f"Warning: Benchmark SMILES '{benchmark_smiles}' could not be parsed or scored.")
        else:
            benchmark_score = benchmark_data["compound_score"]

        # Parse SMILES for the queried compound
        parsed_data = parse_smiles(compound_info["canonical_smiles"], benchmark_score=benchmark_score, weights=weights)

        # Log parsed data for debugging
        logging.debug(f"Parsed data: {parsed_data}")

        # Handle errors in parsed_data
        if "error" in parsed_data:
            flash(f"Error parsing SMILES: {parsed_data['error']}", "danger")
            return render_template("phyto.html")

        return render_template(
            "phyto.html",
            query=query,
            compound_info=compound_info,
            classification=classification,
            parsed_data=parsed_data,
            bioactivities=compound_info.get("bioactivities", []),
            benchmark_smiles=benchmark_smiles,
            benchmark_score=benchmark_score,
        )

    return render_template("phyto.html")

@app.route("/scent", methods=["GET", "POST"])
def scent():
    if request.method == "POST":
        product_type = request.form.get("product_type")
        ingredients = [i.strip() for i in request.form.get("ingredients", "").split(",")]
        strategy = request.form.get("strategy", "random")

        # Validate ingredients against database
        valid_ingredients = []
        with app.app_context():
            for ing in ingredients:
                if Plant.query.filter(Plant.plant_name.ilike(ing)).first():
                    valid_ingredients.append(ing)
        
        if not valid_ingredients:
            flash("No valid ingredients found. Please check your input.", "danger")
            return render_template("scent.html")

        # Fetch and process phytochemicals
        plant_phytochemicals = select_phytochemicals(valid_ingredients, strategy)
        global dataset 
        
        for plant, phytochemicals in plant_phytochemicals.items():
            for phyto in phytochemicals:
                details = fetch_ingredient_details(phyto)
                if details:
                    dataset = pd.concat([dataset, pd.DataFrame([details])], ignore_index=True)

        # Handle missing data
        if dataset.empty:
            flash("No valid bioactive compounds found.", "danger")
            return render_template("scent.html")
        
        dataset['molecular_weight'] = dataset['molecular_weight'].fillna(get_mean_molecular_weight())
        
        def clean_smiles(smiles):
            """Pre-process SMILES to remove invalid characters before parsing."""
            if not isinstance(smiles, str):
                return None
            # Remove characters that might break parsing
            return re.sub(r"[=\(\)]", "", smiles)  

        def safe_parse_smiles(smiles):
            """Safely parse SMILES and return a valid compound score, retrying after cleaning if necessary."""
            try:
                if not smiles:
                    return 0

                # ✅ First attempt: Try parsing the original SMILES
                parsed_data = parse_smiles(smiles)
                if isinstance(parsed_data, dict) and "compound_score" in parsed_data:
                    return parsed_data["compound_score"]

                # ✅ If the first attempt fails, clean the SMILES and retry
                cleaned_smiles = clean_smiles(smiles)
                parsed_data_retry = parse_smiles(cleaned_smiles)

                if isinstance(parsed_data_retry, dict) and "compound_score" in parsed_data_retry:
                    return parsed_data_retry["compound_score"]

                # 🚀 Final Fallback: Estimate compound complexity via atom count
                atom_count_score = fallback_atom_count_score(cleaned_smiles)
                print(f"⚠️ Warning: Using fallback atom count score ({atom_count_score}) for SMILES: {cleaned_smiles}")
                return atom_count_score

            except Exception as e:
                print(f"⚠️ Error parsing SMILES '{smiles}': {e}")
                return 0  # Default value for compatibility

        dataset['compound_score'] = dataset['canonical_smiles'].apply(safe_parse_smiles)

        message = """0.0 - 0.3 → Poor compatibility
                0.3 - 0.6 → Moderate compatibility
                0.6 - 0.8 → Good compatibility
                0.8 - 1.0 → Excellent compatibility"""



        # ✅ Use calculate_compatibility() instead of manually computing compatibility_score
        compatibility_score = calculate_compatibility()

        # Save formulation in database
        user = User.query.filter_by(name=session["user_name"]).first()
        if user:
            new_formulation = UserFormulation(
                user_id=user.id,
                ingredients=", ".join(valid_ingredients),  # Store as comma-separated string
                compatibility_score=compatibility_score
            )
            db.session.add(new_formulation)
            db.session.commit()


        return render_template(
            "scent.html",
            product_type=product_type,
            ingredients=valid_ingredients,
            compatibility_score=f"{compatibility_score:.2f}",
            message=message,
            recommendations=suggest_formulation(product_type,valid_ingredients)
        )

    return render_template("scent.html")

@app.route("/api/plants")
def get_plants():
    search_term = request.args.get('q', '').lower()
    with app.app_context():
        plants = Plant.query.filter(
            Plant.plant_name.ilike(f"%{search_term}%")
        ).with_entities(Plant.plant_name).all()
    return jsonify([plant[0] for plant in plants])



if __name__ == "__main__":
    app.run(debug=True)
