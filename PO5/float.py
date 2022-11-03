# Write a program that will read a .dat file that is composed of:
# 1000000 lines containing floating point numbers
# Convert the numbers to unsigned integers and output how many numbers occur in the file.
# The numbers range from 0 - 4095

# The file is called float.dat
# The output file is called float.out

import sys

# Open the file
f = open("float.dat", "r")

# Create a list to hold the numbers
numbers = []

# Read the file
for line in f:
    # Convert the line to a float
    number = float(line)
    # Convert the float to an integer
    number = int(number)
    # Add the number to the list
    numbers.append(number)

# Close the file
f.close()

# Create a list to hold the counts
counts = []

# Initialize the counts list
for i in range(4096):
    counts.append(0)

# Count the numbers
for number in numbers:
    counts[number] = counts[number] + 1

# Open the output file
f = open("float.out", "w")

# Write the counts to the file
for i in range(4096):
    if counts[i] > 0:
        f.write(str(i) + " " + str(counts[i]) + " times " + " " + str(counts[i]/1000000) + "%")

# Close the file
f.close()

# Run main
if __name__ == "__main__":
    main()
