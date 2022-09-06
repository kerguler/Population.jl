#!/bin/bash -x

# write to file
bootstrap_template()
{
    pandoc --standalone --toc --template=elegant_bootstrap_menu.html tutorial.md -o index.html --metadata title="sPop2: a dynamically-structured matrix population model"
}

# execute it
bootstrap_template