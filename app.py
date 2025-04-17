from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
import torch
from generate import generate_nearby_smiles
from interpolate import interpolate_smiles
from VAE import SimpleTokenizer

app = Flask(__name__)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/molecule_db"
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# JWT Secret Key
app.config["JWT_SECRET_KEY"] = "your_secret_key"

# Get the collections
users_collection = mongo.db.users

# Load tokenizer (used in both generation and interpolation)
simple_tokenizer = SimpleTokenizer()

# ---------------------- 🟢 User Registration Route ----------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Check if user exists
    if users_collection.find_one({"username": username}):
        return jsonify({"error": "User already exists"}), 400

    # Hash password and save to DB
    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    users_collection.insert_one({"username": username, "password": hashed_password, "molecules": []})

    return jsonify({"message": "User registered successfully"}), 201


# ---------------------- 🟢 User Login Route ----------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = users_collection.find_one({"username": username})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    # Generate JWT Token
    access_token = create_access_token(identity=username)
    return jsonify({"message": "Login successful", "access_token": access_token}), 200


# ---------------------- 🟢 Generate Molecules Route ----------------------
@app.route("/generate_smiles", methods=["POST"])
@jwt_required()
def generate_smiles():
    data = request.get_json()
    username = request.json.get("username")  # Extract username from JWT
    smiles = data.get("smiles")
    num_samples = data.get("num_samples", 5)

    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400

    model_path = "beta_tc_vae_model.pth"
    max_len = 172
    temperature = 1.5
    distance_multiplier = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generated_molecules = generate_nearby_smiles(
        model_path, smiles, simple_tokenizer, max_len, num_samples, device, temperature, distance_multiplier
    )

    # Store generated molecules inside the user's document (as an array)
    users_collection.update_one(
        {"username": username}, {"$push": {"molecules": {"$each": generated_molecules}}}
    )

    return jsonify({"input_smiles": smiles, "generated_molecules": generated_molecules})


# ---------------------- 🟢 Interpolation Route ----------------------
@app.route("/interpolate_smiles", methods=["POST"])
@jwt_required()
def interpolate_route():
    data = request.get_json()
    username = request.json.get("username")  # Extract username from JWT
    smiles_1 = data.get("smiles_1")
    smiles_2 = data.get("smiles_2")
    num_steps = data.get("num_steps", 5)

    if not smiles_1 or not smiles_2:
        return jsonify({"error": "Both SMILES strings are required"}), 400

    model_path = "beta_tc_vae_model.pth"
    max_len = 172
    temperature = 1.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interpolated_molecules = interpolate_smiles(
        model_path, smiles_1, smiles_2, simple_tokenizer, max_len, device, num_steps, temperature
    )

    # Store interpolated molecules inside the user's document (as an array)
    users_collection.update_one(
        {"username": username}, {"$push": {"molecules": {"$each": interpolated_molecules}}}
    )

    return jsonify({"input_smiles_1": smiles_1, "input_smiles_2": smiles_2, "interpolated_molecules": interpolated_molecules})


# ---------------------- 🟢 Get User Molecules ----------------------
@app.route("/get_molecules", methods=["GET"])
@jwt_required()
def get_molecules():
    username = request.json.get("username")  # Extract username from JWT
    user = users_collection.find_one({"username": username})

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"username": username, "molecules": user.get("molecules", [])})
@app.route("/get_all_molecules", methods=["GET"])
@jwt_required()
def get_all_molecules():
    # Query all users to gather molecules from each document
    users = users_collection.find()
    
    all_molecules = []
    for user in users:
        all_molecules.extend(user.get("molecules", []))

    return jsonify({"all_molecules": all_molecules})

# ---------------------- 🟢 Run Flask Server ----------------------
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
