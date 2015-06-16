FILENAME="monograph"
REMOVE="rm $FILENAME.aux $FILENAME.lof $FILENAME.log $FILENAME.lot $FILENAME.toc \
$FILENAME.run.xml $FILENAME-blx.bib $FILENAME.bbl $FILENAME.blg $FILENAME.bcf \
$FILENAME.out"

$REMOVE
pdflatex $FILENAME.tex
bibtex $FILENAME.aux
pdflatex $FILENAME.tex
$REMOVE