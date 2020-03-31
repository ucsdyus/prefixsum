# Platform: WINDOWS

NVCCFLAGS = --ptxas-options=-v -O3 -Xcompiler "/w"

APP=pfxsum
OBJECTS = helper.obj pfxsum.obj pfxsum_host.obj pfxsum_naive.obj pfxsum_rs.obj

NVCC = nvcc

%.obj : %.cu
	$(NVCC) -c $(NVCCFLAGS) $@ $<

%.obj : %.cc
	$(NVCC) -c $(NVCCFLAGS) $@ $<

$(APP): $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(NVCCFLAGS)

clean:
	rm -f *.exe *.exp *.lib $(APP) $(OBJECTS)
 