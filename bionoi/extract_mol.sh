# A shell script to extract .mol2 file from ../mols to ../mols_extract.
# The extracted files are files with suffix -1.mol2
mkdir ../mols_extract/
mkdir ../mols_extract/control
mkdir ../mols_extract/heme
mkdir ../mols_extract/nucleotide
mkdir ../mols_extract/steroid
cp ../mols/control/*-1.mol2 ../mols_extract/control/
cp ../mols/heme/*-1.mol2 ../mols_extract/heme/
cp ../mols/nucleotide/*-1.mol2 ../mols_extract/nucleotide/
cp ../mols/steroid/*-1.mol2 ../mols_extract/steroid/
