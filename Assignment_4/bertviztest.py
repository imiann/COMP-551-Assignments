import torch
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view
import matplotlib.pyplot as plt
import numpy as np

def test_bertviz_debug(model_ckpt="bert-base-uncased"):
    """Test BERTViz functionality with detailed debugging and fallback visualization."""
    try:
        # Load model with compatible attention implementation
        print("Loading model...")
        model = AutoModel.from_pretrained(
            model_ckpt,
            output_attentions=True,
            attn_implementation="eager"  # Ensures compatibility for attention visualization
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Model loaded successfully.")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        # Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            print("Padding token not defined, setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

        # Define sample sentences
        sentence_a = "The movie was fantastic and full of surprises."
        sentence_b = "I thoroughly enjoyed the film."

        # Tokenize inputs
        print("Tokenizing inputs...")
        viz_inputs = tokenizer(
            sentence_a,
            sentence_b,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        viz_inputs = {key: val.to(model.device) for key, val in viz_inputs.items()}
        print(f"Tokenized inputs: {viz_inputs}")

        # Forward pass to compute attention
        print("Running forward pass to compute attention...")
        outputs = model(**viz_inputs)
        attention = outputs.attentions
        print("Attention computed successfully.")

        # Debug attention data
        print(f"Attention Shape: {[att.shape for att in attention]}")
        print(f"Sample Attention Weights (Layer 0, Head 0):\n{attention[0][0][0].detach().cpu().numpy()}")

        # Extract tokens and compute sentence B start index
        tokens = tokenizer.convert_ids_to_tokens(viz_inputs["input_ids"][0])
        sentence_b_start = (viz_inputs["token_type_ids"] == 0).sum(dim=1).item()
        print(f"Tokens: {tokens}")
        print(f"Sentence B start index: {sentence_b_start}")

        # Check token alignment with attention matrix
        seq_len = attention[0].shape[-1]
        assert len(tokens) == seq_len, (
            f"Token length {len(tokens)} does not match attention sequence length {seq_len}"
        )

        # Attempt to render visualization
        print("Generating head view visualization...")
        html = head_view(attention, tokens, sentence_b_start, heads=[0])
        if html is None:
            raise ValueError("head_view returned None, visualization could not be generated.")

        # Save visualization to an HTML file
        save_path = "./bertviz_test.html"
        with open(save_path, "w") as f:
            f.write(html)
        print(f"Head view visualization saved to: {save_path}")

    except Exception as e:
        print(f"An error occurred during BERTViz testing: {e}")
        print("Falling back to manual attention visualization...")
        fallback_visualization(attention, tokens)

def fallback_visualization(attention, tokens, output_dir="./Results"):
    """Fallback manual visualization for attention using Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    try:
        for layer in range(len(attention)):  # Iterate through all layers
            for head in range(attention[layer].shape[1]):  # Iterate through all heads
                attention_matrix = attention[layer][0][head].detach().cpu().numpy()

                print(f"Visualizing attention for Layer {layer}, Head {head}...")
                plt.figure(figsize=(10, 8))
                plt.imshow(attention_matrix, cmap='viridis')
                plt.colorbar()
                plt.title(f"Attention Head {head} - Layer {layer}")
                plt.xlabel("Tokens Attended To")
                plt.ylabel("Tokens Attending")
                plt.xticks(range(len(tokens)), tokens, rotation=90)
                plt.yticks(range(len(tokens)), tokens)
                plt.tight_layout()

                # Save the plot as an image
                save_path = os.path.join(output_dir, f"attention_layer{layer}_head{head}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved attention visualization to {save_path}")

        print("All attention visualizations saved successfully.")

    except Exception as e:
        print(f"An error occurred during fallback visualization: {e}")


# Run the test function
test_bertviz_debug()
