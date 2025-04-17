from rdkit import Chem
from rdkit.Chem import Draw
import torch
import torch.nn.functional as F
import warnings
from rdkit import RDLogger
import matplotlib.pyplot as plt 
from VAE import BetaTCVAE, SimpleTokenizer

# Check if running in Jupyter Notebook
try:
    from IPython.display import display  
    JUPYTER_MODE = True
except ImportError:
    from PIL import Image  # Use PIL for non-Jupyter mode
    JUPYTER_MODE = False


def generate_nearby_smiles(model_path, smiles, tokenizer, max_len, num_samples, device, temperature=1.0, distance_multiplier=0.1):
    """
    Generate nearby SMILES strings by perturbing the latent space representation.
    """
    # Load the model
    vocab_size = 6
    embedding_dim = 16
    hidden_dim = 64
    latent_dim = 16
    nhead = 4
    num_layers = 2
    pad_idx = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BetaTCVAE(vocab_size, embedding_dim, hidden_dim, latent_dim, nhead, num_layers, pad_idx, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tokenized_smiles = tokenizer.tokenize(smiles, max_len).unsqueeze(0).to(device)
    embedded = model.embedding(tokenized_smiles)
    encoded = model.encoder(embedded)
    mu = model.mu(encoded)
    log_var = model.log_var(encoded)

    generated_smiles = set()
    i = 0
    while len(generated_smiles) < num_samples:
        random_direction = torch.randn_like(mu)
        random_direction /= torch.norm(random_direction)
        z = mu + distance_multiplier * (i + 1) * random_direction

        decoded = model.decoder(z, encoded)
        out = model.fc_out(decoded)
        out = F.softmax(out / temperature, dim=-1)

        generated_smiles_idx = torch.multinomial(out.squeeze(0), 1).cpu().numpy().flatten()
        generated_smiles_str = ''.join(
            [tokenizer.idx_to_char[min(int(idx), 4)] for idx in generated_smiles_idx if idx != tokenizer.pad_idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(generated_smiles_str)

        if mol is not None:
            generated_smiles.add(generated_smiles_str)

        i += 1  

    return list(generated_smiles)


def generate_2d_molecule(smiles):
    """
    Generate and display a 2D molecular image.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"❌ Could not generate molecule for SMILES: {smiles}")
        return None

    img = Draw.MolToImage(mol, size=(300, 300))

    if JUPYTER_MODE:
        print(f"🖼️ Displaying molecule: {smiles}")  
        plt.imshow(img)  
        plt.axis("off")
        plt.show()
    else:
        img.show()  
        print(f"✅ Displayed molecule: {smiles}")

if __name__ == "__main__":
    # User Input
    input_smiles = input("Enter the input SMILES string: ")
    num_samples = int(input("How many SMILES do you want to generate? "))

    max_len = 172  
    temperature = 2.0
    distance_multiplier = 0.1
    model_path = 'beta_tc_vae_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simple_tokenizer = SimpleTokenizer()
    generated_smiles = generate_nearby_smiles(model_path, input_smiles, simple_tokenizer, max_len, num_samples, device, temperature, distance_multiplier)

    print(f"\nInput SMILES: {input_smiles}")
    print(f"Generated SMILES ({num_samples} unique samples):")
    
    for idx, smiles in enumerate(generated_smiles, start=1):
        print(f"{idx}. {smiles}")

    # Generate and display 2D molecular images
    print("\nGenerating 2D molecular images...\n")
    for smiles in generated_smiles:
        generate_2d_molecule(smiles)
