def load_profanity_list(file_path):
    with open(file_path, 'r') as file:
        profanity_list = [line.strip().lower() for line in file]
    return profanity_list

def load_profanity_list(file_path):
    with open(file_path, 'r') as file:
        profanity_list = [line.strip().lower() for line in file]
    return profanity_list

def apply_profanity_filter(text, profanity_list):
    words = text.lower().split()
    filtered_words = ['*' * len(word) if word in profanity_list else word for word in words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def filter_text_file(input_file_path, output_file_path, profanity_list):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            filtered_line = apply_profanity_filter(line, profanity_list)
            output_file.write(filtered_line + '\n')

def main():
    profanity_list_path = r'C:\Users\thamizh\Desktop\RBG\Obscene filter\en_profane_words.txt'
    input_file_path = r'C:\Users\thamizh\Desktop\RBG\recorded1.txt'  
    output_file_path = r'C:\Users\thamizh\Desktop\RBG\Obscene filter\filtered.txt'  

    profanity_list = load_profanity_list(profanity_list_path)
    filter_text_file(input_file_path, output_file_path, profanity_list)

    print(f"Profanity filtered text saved to: {output_file_path}")

if __name__ == "__main__":
    main()

