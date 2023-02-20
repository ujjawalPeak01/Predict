from transformers import pipeline


def predict_text():
    model = pipeline(
        "text-generation", model="EleutherAI/gpt-neo-125M", device="cuda:0"
    )
    result = model("Once upon a time", do_sample=True, min_length=20)
    return {"prediction": result}
