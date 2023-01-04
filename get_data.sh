# mashup STRING data
cd data/raw/
mkdir mashup_networks
cd mashup_networks
mkdir mashup
cd mashup
wget https://groups.csail.mit.edu/cb/mashup/mashup.tar.gz
tar -xvf mashup.tar.gz
mv data/networks/* ..
cd ..
wget https://stringdb-static.org/download/protein.links.detailed.v11.5/10090.protein.links.detailed.v11.5.txt.gz

# aliase
cd ..
mkdir aliase
cd aliase
wget https://stringdb-static.org/download/protein.aliases.v11.5/9606.protein.aliases.v11.5.txt.gz
wget https://stringdb-static.org/download/protein.aliases.v11.5/10090.protein.aliases.v11.5.txt.gz
gunzip 9606.protein.aliases.v11.5.txt
gunzip 10090.protein.aliases.v11.5.txt

# BioGRID
for org in Homo_sapiens Mus_musculus Saccharomyces_cerevisiae; do
    cd ..
    mv $org ${org}_all
    cd ${org}_all
    mv * ..
    cd ..
    rm -r ${org}_all
done

# GOA
mv GOA goa