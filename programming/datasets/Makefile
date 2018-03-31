SHELL=/bin/bash

#hope this works
all:
	car flower clean

#this make file fetches the datasets for the purpose of the testing of the clustering
flower:
	wget http://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz
	tar -xzvf 102flowers.tgz
	rm 102flowers.tgz
	#mv ./102flowers/jpeg ./flowers
car:
	wget http://imagenet.stanford.edu/internal/car196/bmw10_release.tgz
	tar -xzvf bmw10_release.tgz
	mkdir bmw_subset
	find ./bmw10_release -type f -print0 | xargs -0 mv -t ./bmw_subset
	find ./bmw_subset -type f ! -name "*.jpg" >> purgefiles
	for i in `cat purgefiles`; do rm $i; done
	rm purgefiles
	find ./bmw_subset -type f  -name "*thumb*" >> purgefiles
	for i in `cat purgefiles`; do rm $i; done
	rm purgefiles
	rm -rf bmw10_release

clean:
	rm -rf 102flowers 
	rm bmw10_release.tgz
