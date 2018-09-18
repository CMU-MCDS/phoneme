all: phonemeyhl.pdf

phonemeyhl.pdf: phonemeyhl.tex phonemebib.bib
	pdflatex --shell-escape phonemeyhl.tex
	bibtex phonemeyhl
	pdflatex --shell-escape phonemeyhl.tex
	pdflatex --shell-escape phonemeyhl.tex

CLEAN_FILES = *.aux *.log *.out *.synctex.gz *.bbl *.blg phonemeyhl.pdf
clean:
	rm -f $(CLEAN_FILES)
