SHELL=/bin/bash

all: build clean

build:
	pdflatex -synctex=1 comvi
	pdflatex -synctex=1 comvi
	pdflatex -synctex=1 comvi

clean:
	rm -f *.aux *.bbl *.blg *.glo *.idx *.log *.toc *.out *.synctex.gz *.fls *.fdb_latexmk

cleanall: clean
	rm -f *.ps *.pdf *.dvi *.fls *.fdb_latexmk
