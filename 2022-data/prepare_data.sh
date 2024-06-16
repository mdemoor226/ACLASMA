#!/bin/bash

function Unzip {
	# Specify the directory containing the zip files
	zip_directory=$1

	# Check if the directory exists
	if [ ! -d "$zip_directory" ]; then
	    echo "Directory not found: $zip_directory"
	    exit 1
	fi

	# Change to the specified directory
	cd "$zip_directory" || exit

	# Unzip all zip files
	for zip_file in *.zip; do
	    if [ -f "$zip_file" ]; then
		unzip "$zip_file"
		# If you want to delete the zip file after extracting, uncomment the next line
		 rm "$zip_file"
	    fi
	done

	echo "Unzipping completed."
	cd ../
}

# Check if the directory exists
if [ -d "./dev_data" ]; then
    mkdir ../I-Put-Your-Data-Files-Right-Here
    mv ./dev_data ../I-Put-Your-Data-Files-Right-Here
fi

# Check if the directory exists
if [ -d "./eval_data" ]; then
    mkdir ../I-Put-Your-Data-Files-Right-Here
    mv ./eval_data ../I-Put-Your-Data-Files-Right-Here
fi

mkdir Prepare
cd Prepare

# Download Dev Data...
echo "Downloading Dev Data..."
wget https://zenodo.org/api/records/6355122/files-archive

# Prepare Dev Data...
mv files-archive files-archive.zip
unzip files-archive.zip
rm files-archive.zip
mkdir dev_data
mv *.zip dev_data
Unzip "dev_data"

# Download Eval (The Extra Train) Data...
echo "Downloading Eval/Extra Training Data..."
wget https://zenodo.org/api/records/6462969/files-archive

# Prepare Eval Data...
mv files-archive files-archive.zip
unzip files-archive.zip
rm files-archive.zip
mkdir eval_data
mv *.zip eval_data
Unzip "eval_data"

# Donwload Test Data...
echo "Downloading Test Data..."
wget https://zenodo.org/api/records/6586456/files-archive

# Prepare Test Data...
mv files-archive files-archive.zip
unzip files-archive.zip
rm files-archive.zip
mkdir test_data
mv *.zip test_data
Unzip "test_data"

# Move Test Files Into Eval Folder
cd ./test_data
for subdirectory in */; do
    if [ -d "$subdirectory" ]; then
        mv $subdirectory/test ../eval_data/$subdirectory
    fi
done    

git clone "https://github.com/Kota-Dohi/dcase2022_evaluator"
mv dcase2022_evaluator ../eval_data/

# Cleanup
cd ../
rm -rf test_data/
mv * ../
cd ../
rm -rf Prepare


