all: phoneme.pdf

phoneme.pdf: phoneme.tex phonemebib.bib
	pdflatex --shell-escape phoneme.tex
	bibtex phoneme
	pdflatex --shell-escape phoneme.tex
	pdflatex --shell-escape phoneme.tex

CLEAN_FILES = *.aux *.log *.out *.synctex.gz *.bbl *.blg phoneme.pdf
clean:
	rm -f $(CLEAN_FILES)
