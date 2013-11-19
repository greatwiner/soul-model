dlvl = ./.
include $(dlvl)/Makefile.in
help:
	@ echo "Structure OUtput Layer Language Model Toolkit"

all: env makedir config ioFile tensor text module model command script test

env:
	. /people/dokhanh/.bash_profile

makedir:
	mkdir -p $(DIR_INSTALL)
	mkdir -p $(CLIB_PATH)
	mkdir -p $(CBIN_PATH)
	mkdir -p $(CINCLUDE_PATH)
	mkdir -p $(CSCRIPT_PATH)
	
config: 
	(cd package/config;\
	make all;)
	
ioFile: 
	(cd package/ioFile;\
	make all;)
	
tensor: 
	(cd package/tensor;\
	make all;)
text: 
	(cd package/text;\
	make all;)
module: 
	(cd package/module;\
	make all;)
model: 
	(cd package/model;\
	make all;)
command: 
	(cd package/command;\
	make all;)
	
cleanInSource:
	(cd package/ioFile;\
	rm -f *.o;)
	(cd package/tensor;\
	rm -f *.o;)
	(cd package/text;\
	rm -f *.o;)
	(cd package/module;\
	rm -f *.o;)
	(cd package/model;\
	rm -f *.o;)	
	(cd package/command;\
	rm -f *.exe;)
	(cd package/test;\
	rm -f *.exe;)

clean:
	rm -r $(DIR_INSTALL)/*

test:
	(cd package/test;\
	make all;)	
forwardTrainTest:
	$(CBIN_PATH)forwardTrainTest.exe
script:
	(cd package/script;\
	make all;)
