convert_and_delet(){
    files=$(ls *.mp3)
    for x in $files
    do
        target="${x%.*}"
        ffmpeg -i $x "${target}.wav"
        rm -f $x
    done
}

cd test || exit
convert_and_delet
echo "test data all converted!"
cd - || exit
cd train || exit
convert_and_delet
cd - || exit