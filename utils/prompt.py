
def prompt(args, question):
    if "WebQA" == args.datasets:
            # prompt_template = "Question: " + question + "\nAnswer the question using a single sentence."
            prompt_template = question
    else:
        prompt_template = (
            f"{question}\nAnswer the question using a single word or phrase."
        )
    return prompt_template

