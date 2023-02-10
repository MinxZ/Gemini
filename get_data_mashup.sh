# mashup STRING data
cd data/
mkdir raw
mkdir results
cd raw
mkdir mashup_networks
cd mashup_networks
mkdir mashup
cd mashup
wget https://groups.csail.mit.edu/cb/mashup/mashup.tar.gz
tar -xvf mashup.tar.gz
mv data/networks/* ..
cd ..
wget https://stringdb-static.org/download/protein.links.detailed.v11.5/10090.protein.links.detailed.v11.5.txt.gz
gunzip 10090.protein.links.detailed.v11.5.txt
cd ..
