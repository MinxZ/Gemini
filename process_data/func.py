"""
Mingxin Zhang
Help functions
"""

def textread(filename, astype=str):
    output = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            i = astype(line.strip())
            output.append(i)
    return output
