import re
import argparse

def extract_data_from_line(line):
    match = re.match(r"([A-Za-z0-9]+)\s+(.+)", line)
    if match:
        formula = match.group(1)
        data = match.group(2).split('|')
        data = [tuple(map(float, item.split())) for item in data]
        return formula, data
    return None, None

def process_file(filename):
    x_values = []
    y_values = []
    intensities = []
    
    with open(filename, 'r') as file:
        for line in file:
            formula, data = extract_data_from_line(line.strip())
            if data:
                for x, y, intensity in data:
                    x_values.append(x)
                    y_values.append(y)
                    intensities.append(intensity)

    return x_values, y_values, intensities

def calculate_ranges(values):
    return min(values), max(values)

def main(filename):
    x_values, y_values, intensities = process_file(filename)
    
    x_range = calculate_ranges(x_values)
    y_range = calculate_ranges(y_values)
    intensity_range = calculate_ranges(intensities)

    print(f"X values range: {x_range}")
    print(f"Y values range: {y_range}")
    print(f"Intensity values range: {intensity_range}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NMR data from a file.")
    parser.add_argument('filename', type=str, help="Path to the NMR data file.")
    
    args = parser.parse_args()

    main(args.filename)
