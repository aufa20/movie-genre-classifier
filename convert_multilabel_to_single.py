def convert_multilabel_to_single(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            parts = line.strip().split()
            labels = [word for word in parts if word.startswith("__label__")]
            text_start = len(labels)
            
            if not labels:
                continue  # skip lines without labels

            # Pick the first label (you can change this logic if needed)
            first_label = labels[0]
            text = " ".join(parts[text_start:])
            outfile.write(f"{first_label} {text}\n")


if __name__ == "__main__":
    input_path = "train_multilabel.txt"     # original file with multi-labels
    output_path = "train_singlelabel.txt"   # new file for FastText single-label training

    convert_multilabel_to_single(input_path, output_path)
    print(f"✅ Done! Converted to single-label format at: {output_path}")
