# Simple SOP to use STRIPE
~ Raghav, Oct 2026

1. Create raw (binary) file from SIGPROC filterbank (i.e. remove metadata header from the start of file)
	- Get the header size of the filterbank using `header` command

	 `header /path/to/filterbank`
	
	- Use Disk Destroyer (`dd`) command to make header and raw file from filterbank
	NOTE: BE CAREFUL ABOUT THE INPUT (`if`) and OUTPUT (`of`) FILE NAMES, AS IT CAN LITERALLY DESTROY YOUR FILE ON THE DISK

	```bash
	  $ dd if=FILENAME.fil of=FILENAME.fil.hdr bs=1 count=<header_size> # Makes header file 
	  $ dd if=FILENAME.fil of=FILENAME.fil.raw bs=1M skip=<header_size> iflag=skip_bytes # Makes raw file
    ```
2. Execute RFI mitigation using STRIPE 

	`./stripe <binary_file_path> <block_size> <threshold> <width>`

	You can use block size of 800, threshold of 4 and width of 0.5
	This will create '*_stripe.raw' file in the same location as the input raw file

3. Make the filterbank from STRIPE mitigated raw file
	
	`cat FILENAME.fil.hdr FILENAME.fil_stripe.raw > FILENAME.fil_stripe.raw.fil`
	

