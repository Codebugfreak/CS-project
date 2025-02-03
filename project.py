from datetime import datetime

def happy_birthday(user_name):
    return f"Happy birthday, {user_name}!"


def normalize_path(path):
    """Normalize file paths to use forward slashes for web compatibility."""
    if path:
        return path.replace("\\", "/")
    return None


