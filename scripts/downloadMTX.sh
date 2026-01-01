# Definizione delle matrici (Nome Cartella/Nome Matrice)
# Note: thermal2 è in Schmid, Flan_1565 in Janna, cit-Patents in Patents, soc-LiveJournal1 in SNAP
declare -A MATRICES
MATRICES["cit-Patents"]="Patents"
MATRICES["soc-LiveJournal1"]="SNAP"
MATRICES["thermal2"]="Schmid"
MATRICES["Flan_1565"]="Janna"

for MAT in "${!MATRICES[@]}"; do
    GROUP=${MATRICES[$MAT]}
    if [ ! -f "$MATRIX_DIR/$MAT.mtx" ]; then
        echo "Downloading $MAT..."
        # Download dal mirror ufficiale SuiteSparse
        wget -q "https:/suitesparse-collection-website.herokuapp.com/MM/$GROUP/$MAT.tar.gz" -P "$MATRIX_DIR"
        
        # Estrazione (tar -xzf gestisce sia tar che gzip in un colpo solo)
        tar -xzf "$MATRIX_DIR/$MAT.tar.gz" -C "$MATRIX_DIR"
        
        # Sposta il file mtx ed elimina il superfluo
        mv "$MATRIX_DIR/$MAT/$MAT.mtx" "$MATRIX_DIR/"
        rm -rf "$MATRIX_DIR/$MAT" "$MATRIX_DIR/$MAT.tar.gz"
        
        echo "$MAT.mtx scaricata e pronta."
    else
        echo "$MAT.mtx già presente, salto il download."
    fi
done