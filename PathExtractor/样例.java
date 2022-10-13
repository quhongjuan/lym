//  try to find layout xml file in selected element
public static PsiFile findLayoutResource(PsiElement element){
	if(element == null){
			return null ;}
	if(!(element instanceof PsiIdentifier)){
		return null;}
	PsiElement layout=element.getParent().getFirstChild();
	if(layout == null){
		return null;}
	if(!STR_.equals(layout.getText())){
			return null;}
	Project project=element.getProject();
	String name=String.format(STR_,element.getText());
	return resolveLayoutResourceFile(element,project,name);
}
public int FindProc(String id){
	int i=0; 
	while(i<procs.size()){
		ProcedureEntry pe=procs.elementAt(i);
		if(pe.name.equals(id))
			return i;
		i=i+1;}
	return i;}


// Path1: ['FindProc','String id','int i=NUM','while(i<procs.size())','ProcedureEntry pe=procs.elementAt(i)','if(pe.name.equals(id))','returni']
// Path2: ['FindProc','String id','int i=NUM','while(i<procs.size())','ProcedureEntry pe=procs.elementAt(i)','if(pe.name.equals(id))','i=i+NUM']
// Path3: ['FindProc','String id','int i=NUM','while(i<procs.size())','return i']