# ------------------------------------------
# Matrix Directories and Download
# ------------------------------------------
# Note: thermal2 is in Schmid, Flan_1565 in Janna, 
# cit-Patents in Patents, soc-LiveJournal1 in SNAP
declare -A MATRICES
MATRICES["cit-Patents"]="Patents"
MATRICES["soc-LiveJournal1"]="SNAP"
MATRICES["thermal2"]="Schmid"
MATRICES["Flan_1565"]="Janna"

for MAT in "${!MATRICES[@]}"; do
    GROUP=${MATRICES[$MAT]}
    if [ ! -f "$MATRIX_DIR/$MAT.mtx" ]; then
        echo "Downloading $MAT..."
        # Download from the official SuiteSparse mirror
        wget -q "https://suitesparse-collection-website.herokuapp.com/MM/$GROUP/$MAT.tar.gz" -P "$MATRIX_DIR"

        # Extraction (tar -xzf handles both decompression and extraction)
        tar -xzf "$MATRIX_DIR/$MAT.tar.gz" -C "$MATRIX_DIR"
        
        # Move .mtx file to MATRIX_DIR and clean up temporary files
        mv "$MATRIX_DIR/$MAT/$MAT.mtx" "$MATRIX_DIR/"
        rm -rf "$MATRIX_DIR/$MAT" "$MATRIX_DIR/$MAT.tar.gz"
        
        echo "$MAT.mtx downloaded and ready."
    else
        echo "$MAT.mtx is already present, skipping download."
    fi
done