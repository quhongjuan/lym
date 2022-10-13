public int FindProc(String id){
	int i=0; 
	while(i<procs.size()){
		ProcedureEntry pe=procs.elementAt(i);
		if(pe.name.equals(id))
			return i;
		i=i+1;}
	return i;}


Path1: ['FindProc','String id','int i=NUM','while(i<procs.size())','ProcedureEntry pe=procs.elementAt(i)','if(pe.name.equals(id))','returni']
Path2: ['FindProc','String id','int i=NUM','while(i<procs.size())','ProcedureEntry pe=procs.elementAt(i)','if(pe.name.equals(id))','i=i+NUM']
Path3: ['FindProc','String id','int i=NUM','while(i<procs.size())','return i']