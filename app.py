from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from bcrypt import hashpw, gensalt, checkpw
import os
from datetime import datetime, timezone
from project import happy_birthday, normalize_path
from deep_dream import generate_visual, generate_animation
from deep_dream import LAYER_METADATA
from phyto import fetch_phytochemical_info, classify_phytochemical, parse_smiles
from scent import fetch_ingredient_details, suggest_formulation, calculate_compatibility



app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLAlchemy configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'users.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    password = db.Column(db.String(200), nullable=False)  # Increased length for hashed passwords
    birthday = db.Column(db.String(10), nullable=False)
    profile_pic = db.Column(db.String(150), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

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

    # Normalize the profile picture path for rendering
    profile_pic_url = normalize_path(user.profile_pic)

    # Render the profile template
    return render_template(
        'profile.html',
        user_name=user.name,
        email=user.email,
        birthday=user.birthday,
        bio=user.bio,
        profile_pic=profile_pic_url or "https://via.placeholder.com/150", # Default profile pic
        created_at=user.created_at,
        updated_at=user.updated_at
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
            animation = request.form.get("animation", "false").lower() == "true"

            # Generate the deep dream image
            output_path = generate_visual(
                image_path,
                model_name=model_name,
                layer_name=layer_name,
                gradient_multiplier=gradient_multiplier,
                iterations=iterations,
                apply_filter=apply_filter,
                animation=animation
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
    return redirect(url_for('deep_dream'))


# Register route
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
            os.makedirs(upload_dir, exist_ok=True)
            profile_pic_path = os.path.join('static/uploads', profile_pic.filename)
            profile_pic.save(profile_pic_path)

        # Add user to the database
        new_user = User(
            name=user_name,
            email=email,
            password=hashed_password.decode('utf-8'),  # Store as string in the database
            birthday=birthday,
            bio=bio,
            profile_pic=profile_pic_path
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful.", "success")
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
            return redirect(url_for('home'))

        flash("Invalid username or password.", "danger")
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_name', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

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

        # Classify compound and parse SMILES
        classification = classify_phytochemical(compound_info["canonical_smiles"])
        parsed_data = parse_smiles(compound_info["canonical_smiles"])

        return render_template(
            "phyto.html",
            query=query,
            compound_info=compound_info,
            classification=classification,
            parsed_data=parsed_data,
        )

    return render_template("phyto.html")    

# Create your scent
@app.route("/scent", methods=["GET", "POST"])
def scent():
    if request.method == "POST":
        product_type = request.form.get("product_type")
        ingredients = request.form.getlist("ingredients")
        
        # Fetch ingredient details from PubChem
        ingredient_details = [fetch_ingredient_details(ingredient) for ingredient in ingredients if ingredient]

        # Suggest optimal formulation
        suggested_formulation = suggest_formulation(product_type, ingredients)

        # Calculate compatibility
        compatibility_score = calculate_compatibility(ingredients)

        return render_template(
            "scent.html",
            product_type=product_type,
            ingredients=ingredients,
            ingredient_details=ingredient_details,
            suggested_formulation=suggested_formulation,
            compatibility_score=compatibility_score,
        )
    return render_template("scent.html")

@app.route("/scent/phytochemicals", methods=["GET", "POST"])
def scent_phytochemicals():
    if request.method == "POST":
        phytochemicals = request.form.getlist("phytochemicals")
        
        # Fetch details of each phytochemical
        phytochemical_details = [fetch_ingredient_details(name) for name in phytochemicals]
        
        return render_template(
            "scent_phytochemicals.html",
            phytochemicals=phytochemicals,
            phytochemical_details=phytochemical_details,
        )
    return render_template("scent_phytochemicals.html")



if __name__ == "__main__":
    app.run(debug=True)
