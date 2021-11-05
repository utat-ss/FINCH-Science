wavenumber = []
spec_line_intensity = []

with open('methane-line-by-line.par') as methane:
    for row in methane:
        wavenumber.append(float(row[4:15]))
        spec_line_intensity.append(float(row[17:26]))

print(wavenumber)
print(spec_line_intensity)