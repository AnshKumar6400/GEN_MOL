import torch
from rdkit import Chem
import torch.nn.functional as F
import warnings
from rdkit.Chem import Draw
from rdkit import RDLogger
from VAE import BetaTCVAE, SimpleTokenizer
import matplotlib.pyplot as plt  
from PIL import Image  

# Check if running in Jupyter Notebook
try:
    from IPython.display import display  
    JUPYTER_MODE = True
except ImportError:
    JUPYTER_MODE = False

def interpolate_smiles(model_path, smiles_1, smiles_2, tokenizer, max_len, device, num_steps=5, temperature=1.0):
    """
    Interpolate between two SMILES strings in the latent space.

    Args:
        model_path (str): Path to the saved model.
        smiles_1 (str): First SMILES string.
        smiles_2 (str): Second SMILES string.
        tokenizer (SimpleTokenizer): Tokenizer object to tokenize SMILES.
        max_len (int): Maximum length of the tokenized sequences.
        device (torch.device): Device to run the model on.
        num_steps (int): Number of interpolation steps.
        temperature (float): Temperature for sampling.

    Returns:
        list of str: List of interpolated SMILES strings. 
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

    tokenized_smiles_1 = tokenizer.tokenize(smiles_1, max_len).unsqueeze(0).to(device)
    tokenized_smiles_2 = tokenizer.tokenize(smiles_2, max_len).unsqueeze(0).to(device)

    embedded_1 = model.embedding(tokenized_smiles_1)
    encoded_1 = model.encoder(embedded_1)
    mu_1 = model.mu(encoded_1)

    embedded_2 = model.embedding(tokenized_smiles_2)
    encoded_2 = model.encoder(embedded_2)
    mu_2 = model.mu(encoded_2)

    average_encoded = (encoded_1 + encoded_2) / 2

    interpolated_smiles = []
    for i in range(num_steps + 1):
        alpha = i / num_steps
        z = mu_1 * (1 - alpha) + mu_2 * alpha
        decoded = model.decoder(z, average_encoded)
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
            interpolated_smiles.append(generated_smiles_str)

    return list(set(interpolated_smiles))
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
    # Example usage
    smiles_1 = input("🔹 Enter the first SMILES string: ")
    smiles_2 = input("🔹 Enter the second SMILES string: ")
    num_steps = int(input("🔹 How many interpolation steps? "))

    max_len = 172  
    temperature = 1.5
    model_path = 'beta_tc_vae_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simple_tokenizer = SimpleTokenizer()
    interpolated_smiles = interpolate_smiles(model_path, smiles_1, smiles_2, simple_tokenizer, max_len,
                                             device, num_steps, temperature)

    print(f"\n🔹 First Input SMILES: {smiles_1}")
    print(f"🔹 Second Input SMILES: {smiles_2}")
    print(f"🔹 Interpolated SMILES ({num_steps} steps):")
    
    for idx, smiles in enumerate(interpolated_smiles, start=1):
        print(f"{idx}. {smiles}")

    # **Generate and display 2D molecular images**
    print("\n🎨 Generating 2D molecular images...\n")
    for smiles in interpolated_smiles:
        generate_2d_molecule(smiles)