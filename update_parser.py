"""
Helper script to update the argument parser in visualize_tsp_predictions.py
to include the show_all_visualizations option.
"""
import re

def update_argparse():
    # Path to the visualization script
    file_path = '/home/cailtsptest/tsp/benchmarking-gnns/visualize_tsp_predictions.py'
    
    try:
        # Read the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Find the parse_args function to modify
        parser_pattern = r'def parse_args\(\):\n.*?parser = argparse\.ArgumentParser.*?\n'
        parser_match = re.search(parser_pattern, content, re.DOTALL)
        
        if parser_match:
            # Find the end of the argument definitions
            args_pattern = r'(def parse_args.*?args = parser\.parse_args\(\))'
            args_match = re.search(args_pattern, content, re.DOTALL)
            
            if args_match:
                # Insert our new argument before the parse_args() call
                updated_section = args_match.group(1).replace(
                    'args = parser.parse_args()',
                    "    parser.add_argument('--show_all_visualizations', action='store_true', help=\"Show additional visualizations\")\n\n    args = parser.parse_args()"
                )
                
                # Replace the section
                updated_content = content.replace(args_match.group(1), updated_section)
                
                # Write the updated content back to the file
                with open(file_path, 'w') as file:
                    file.write(updated_content)
                
                print(f"Successfully updated {file_path}")
                return True
            
        print(f"Could not find the appropriate sections to update in {file_path}")
        return False
    
    except Exception as e:
        print(f"Error updating file: {str(e)}")
        return False

if __name__ == "__main__":
    update_argparse()
