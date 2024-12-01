import torch
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view
from IPython.display import display, HTML

def test_bertviz(model_ckpt="bert-base-uncased"):
    """Test BERTViz functionality with a simple input."""
    try:
        # Load model with compatible attention implementation
        model = AutoModel.from_pretrained(
            model_ckpt,
            output_attentions=True,
            attn_implementation="eager"  # Ensures compatibility for attention visualization
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        # Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

        # Define sample sentences
        sentence_a = "The movie was fantastic and full of surprises."
        sentence_b = "I thoroughly enjoyed the film."

        # Tokenize inputs
        viz_inputs = tokenizer(
            sentence_a,
            sentence_b,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        viz_inputs = {key: val.to(model.device) for key, val in viz_inputs.items()}

        # Forward pass to compute attention
        outputs = model(**viz_inputs)
        attention = outputs.attentions

        # Extract tokens and compute sentence B start index
        tokens = tokenizer.convert_ids_to_tokens(viz_inputs["input_ids"][0])
        sentence_b_start = (viz_inputs["token_type_ids"] == 0).sum(dim=1).item()

        # Render visualization
        print("Generating head view visualization...")
        html = head_view(attention, tokens, sentence_b_start, heads=[8])

        # Save visualization to an HTML file
        save_path = "./bertviz_test.html"
        with open(save_path, "w") as f:
            f.write(html)
        print(f"Head view visualization saved to: {save_path}")

    except Exception as e:
        print(f"An error occurred during BERTViz testing: {e}")

# Run the test function
test_bertviz()
