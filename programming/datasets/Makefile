SHELL=/bin/bash

#hope this works
all:
	flower clean

#this make file fetches the datasets for the purpose of the testing of the clustering
flower:
	wget http://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz
	tar -xzvf 102flowers.tgz
	rm 102flowers.tgz
	#mv ./102flowers/jpeg ./flowers

clean:
	rm -rf 102flowers 
