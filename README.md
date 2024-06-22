
# Table of Contents

1.  [Reference](#org6922805)
2.  [Running](#org2d83da5)
3.  [Test](#orgab9f375)
4.  [Getting the data](#org9bdb527)



<a id="org6922805"></a>

# Reference

-   ISDT: <https://github.com/UniversalDependencies/UD_Italian-ISDT>
    -   used commit `5bb0bf3`
-   <https://gist.github.com/syllog1sm/10343947>


<a id="org2d83da5"></a>

# Running

In general use with:

    python parser.py --train wsj_train.dep
    python parser.py --test wsj_train.pos wsj_test.dep
    python parser.py --query 'test sentence'


<a id="orgab9f375"></a>

# Test

To test you can run:

    python parser.py -h
    python parser.py --train data/it_isdt-ud-train.conll 15
    python parser.py --test data/heldout.pos data/it_isdt-ud-test.conll
    python parser.py -q 'essere guerra con il nemico " .'


<a id="org9bdb527"></a>

# Getting the data

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
