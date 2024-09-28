import os

def consolidate_files(folder_path, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            if filename.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Write the filename and relative path as a comment in Markdown format
                outfile.write(f"<!-- File: {filename} -->\n<!-- Path: {file_path} -->\n")
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read() + '\n\n')  # Add a newline between files

# Example usage
consolidate_files('/Users/prompt/Documents/Github/local_vision', 'consolidated_output.md')