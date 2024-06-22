
# Table of Contents



-   ISDT: <https://github.com/UniversalDependencies/UD_Italian-ISDT>
    -   used commit `5bb0bf3`

In general run with:

    python parser.py wsj_train.dep wsj_train.pos wsj_test.dep

To test run:

    python parser.py data/it_isdt-ud-train.conll data/heldout.pos data/it_isdt-ud-test.conll

Obtain `wsj_train.dep` <sup><a id="fnr.1" class="footref" href="#fn.1" role="doc-backlink">1</a></sup>:

    for f in $1/*.mrg; do
        echo $f
        grep -v CODE $f > "$f.2"
        out="$f.dep"
        java -mx800m -cp "$scriptdir/*:" edu.stanford.nlp.trees.EnglishGrammaticalStructure \
    -treeFile "$f.2" -basic -makeCopulaHead -conllx > $out
    done

Convert to `conll-x` format<sup><a id="fnr.2" class="footref" href="#fn.2" role="doc-backlink">2</a></sup>:

    perl conllu_to_conllx.pl < file.conllu > file.conll

Extract word/tag tuples for the test sentences:

    ../tools/extract-pos it_isdt-ud-test.conllu


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <https://explosion.ai/blog/parsing-english-in-python>

<sup><a id="fn.2" href="#fnr.2">2</a></sup> <https://github.com/UniversalDependencies/tools>
