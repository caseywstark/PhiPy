from fowler import Element, Cluster

indium = Element('In', 'Indium', 49)
potassium = Element('K', 'Potassium', 19)

potassium_39 = Cluster(potassium, 39)
indium_22 = Cluster(indium, 22)
indium_25 = Cluster(indium, 25)

potassium_39_data = {
    'cluster': potassium_39,
    'photon_energies': [2.97382151, 2.97382151, 2.97382151, 2.97382151,
        2.97124714, 2.97124714, 2.97124714, 2.97124714, 2.968672769,
        2.968672769, 2.968672769, 2.966098398, 2.966098398, 2.963524027,
        2.963524027, 2.960949657, 2.960949657, 2.960949657, 2.960949657,
        2.960949657, 2.958375286, 2.958375286, 2.958375286, 2.955800915,
        2.955800915, 2.955800915, 2.955800915, 2.955800915, 2.955800915,
        2.955800915, 2.953226545, 2.950652174, 2.948077803, 2.948077803,
        2.948077803, 2.945503432, 2.945503432, 2.942929062, 2.942929062,
        2.940354691, 2.940354691, 2.940354691, 2.93778032, 2.93778032,
        2.93520595, 2.93520595, 2.93520595, 2.93520595, 2.932631579,
        2.932631579, 2.930057208, 2.930057208, 2.927482838, 2.927482838,
        2.924908467, 2.924908467, 2.922334096, 2.922334096, 2.919759725,
        2.917185355, 2.917185355, 2.917185355, 2.914610984, 2.912036613,
        2.912036613, 2.909462243, 2.909462243, 2.909462243, 2.906887872,
        2.906887872, 2.904313501, 2.904313501, 2.904313501, 2.90173913,
        2.89916476, 2.896590389, 2.894016018, 2.894016018, 2.894016018,
        2.891441648, 2.888867277, 2.888867277, 2.888867277, 2.886292906,
        2.886292906, 2.883718535, 2.883718535, 2.881144165, 2.881144165,
        2.881144165, 2.878569794, 2.878569794, 2.875995423, 2.875995423,],
    'yields': [0.761312217, 0.737556561, 0.7239819, 0.68438914, 0.737556561,
        0.725113122, 0.71040724, 0.699095023, 0.690045249, 0.678733032,
        0.666289593, 0.71040724, 0.695701357, 0.647058824, 0.631221719,
        0.625565611, 0.610859729, 0.597285068, 0.585972851, 0.569004525,
        0.623303167, 0.606334842, 0.595022624, 0.614253394, 0.599547511,
        0.587104072, 0.574660633, 0.561085973, 0.547511312, 0.533936652,
        0.516968326, 0.497737557, 0.484162896, 0.472850679, 0.459276018,
        0.455882353, 0.441176471, 0.451357466, 0.437782805, 0.39479638,
        0.383484163, 0.371040724, 0.385746606, 0.372171946, 0.395927602,
        0.384615385, 0.373303167, 0.360859729, 0.358597285, 0.347285068,
        0.339366516, 0.325791855, 0.279411765, 0.265837104, 0.285067873,
        0.269230769, 0.270361991, 0.257918552, 0.228506787, 0.239819005,
        0.228506787, 0.214932127, 0.234162896, 0.21719457, 0.191176471,
        0.20361991, 0.188914027, 0.174208145, 0.190045249, 0.178733032,
        0.187782805, 0.156108597, 0.14479638, 0.153846154, 0.153846154,
        0.150452489, 0.180995475, 0.168552036, 0.15158371, 0.119909502,
        0.110859729, 0.099547511, 0.088235294, 0.117647059, 0.10520362,
        0.095022624, 0.082579186, 0.081447964, 0.070135747, 0.056561086,
        0.083710407, 0.071266968, 0.097285068, 0.085972851,],
}

indium_22_data = {
    'cluster': indium_22,
    'photon_energies': [5.08E+00, 5.10E+00, 5.12E+00, 5.14E+00, 5.16E+00, 
        5.18E+00, 5.20E+00, 5.22E+00, 5.24E+00, 5.26E+00, 5.28E+00, 5.31E+00,
        5.33E+00, 5.35E+00, 5.38E+00, 5.40E+00, 5.42E+00, 5.45E+00, 5.47E+00,
        5.49E+00, 5.52E+00, 5.54E+00, 5.57E+00, 5.60E+00, 5.64E+00, 5.62E+00,
        5.67E+00, 5.69E+00, 5.72E+00, 5.74E+00, 5.77E+00, 5.83E+00, 5.80E+00,
        5.88E+00,],
    'yields': [2.30E-02, 2.30E-02, 2.81E-02, 2.80E-02, 2.65E-02, 3.37E-02,
        4.32E-02, 5.21E-02, 5.58E-02, 7.41E-02, 7.99E-02, 7.51E-02, 9.14E-02,
        1.14E-01, 1.18E-01, 1.50E-01, 2.28E-01, 2.09E-01, 2.57E-01, 2.40E-01,
        2.95E-01, 3.39E-01, 3.74E-01, 4.03E-01, 4.65E-01, 5.35E-01, 5.39E-01,
        6.60E-01, 6.60E-01, 7.49E-01, 8.07E-01, 8.24E-01, 8.67E-01, 9.37E-01,]
}

indium_25_data = {
    'cluster': indium_25,
	'photon_energies': [5.06E+00, 5.08E+00, 5.10E+00, 5.12E+00, 5.14E+00,
	    5.16E+00, 5.18E+00, 5.20E+00, 5.22E+00, 5.24E+00, 5.26E+00, 5.28E+00,
	    5.30E+00, 5.32E+00, 5.34E+00, 5.36E+00, 5.38E+00, 5.40E+00, 5.42E+00,
	    5.45E+00, 5.49E+00, 5.47E+00, 5.52E+00, 5.54E+00, 5.57E+00, 5.60E+00,
	    5.64E+00, 5.67E+00, 5.70E+00, 5.62E+00, 5.72E+00, 5.75E+00, 5.77E+00,
	    5.80E+00, 5.83E+00, 5.86E+00, 5.89E+00, 5.94E+00, 5.97E+00, 5.91E+00,
	    5.80E+00, 5.83E+00, 5.86E+00, 5.89E+00, 5.94E+00, 5.97E+00, 5.91E+00,],
	'yields': [2.78E-02, 2.81E-02, 3.55E-02, 4.30E-02, 4.88E-02, 4.96E-02,
	    5.93E-02, 6.97E-02, 8.10E-02, 9.61E-02, 1.01E-01, 1.10E-01, 1.14E-01,
	    1.28E-01, 1.44E-01, 1.67E-01, 1.78E-01, 2.59E-01, 2.87E-01, 2.90E-01,
	    2.96E-01, 3.53E-01, 3.59E-01, 3.67E-01, 3.81E-01, 4.24E-01, 5.12E-01,
	    5.80E-01, 5.79E-01, 6.41E-01, 6.55E-01, 6.55E-01, 7.60E-01, 7.69E-01,
	    7.76E-01, 8.80E-01, 8.75E-01, 9.26E-01, 9.44E-01, 9.67E-01, 7.69E-01,
	    7.76E-01, 8.80E-01, 8.75E-01, 9.26E-01, 9.44E-01, 9.67E-01,]
}

data = {
    'Potassium-39': potassium_39_data,
    'Indium-22': indium_22_data,
    'Indium-25': indium_25_data,
}