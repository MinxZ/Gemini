Gemini implementation in Python

Dataset directory structure
```
-data
    -raw
        -Saccharomyces_cerevisiae
            -Co-expression.Abbott-Pronk-2008.txt
            ...
        -Home_sapiens
        -Mus_musculus
        -goa
            -GOA_human.csv
            -GOA_mouse.csv
            -GOA_yeast.csv
        -aliase
            -10090.protein.aliases.v11.5.txt
            -9606.protein.aliases.v11.5.txt
        -mashup_networks
            -human
            -yeast
```

`sh data_preprocess.sh`

`sh main.sh`
```

You can put embedding data under data directory and naming it like {method}_{org}_{annotype}_{demision}.npy like gemini_yeast_go_500.npy to run other methods cross-vaidation.

Download formatted data from: https://drive.google.com/drive/folders/1l0kK5MqiNlWAMSXrNgN4FJphHIzWraGg
