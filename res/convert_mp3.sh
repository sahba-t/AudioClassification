convert_and_delet(){
    files=$(ls *.mp3)
    for x in $files
    do
	target="${x%.*}"
	ffmpeg -i $x "${target}.wav"
	rm -f $x
    done
}
cd test
convert_and_delet
echo "test data all converted!"
cd -
cd train
convert_and_delet
