from graphviz import Digraph

dot = Digraph(comment='Project Flow Diagram', format='png')
dot.attr(rankdir='TB', size='10,10')

# Nodes
dot.node('Query', 'Query Input\n(Doping/Match/Cyber)')
dot.node('Task', 'Task Agent\n(Intent Parsing\n90% doping, 70% cyber)')
dot.node('Doping', 'Doping Agent')
dot.node('Match', 'Match-Fixing Agent')
dot.node('Cyber', 'Cyber Agent')
dot.node('D_Pre', 'Preprocessing\n(StandardScaler)', shape='box')
dot.node('D_Model', 'Model\n(XGBoost)', shape='box')
dot.node('D_SHAP', 'SHAP Explainer\n(shap)', shape='box')
dot.node('M_Pre', 'Preprocessing\n(StandardScaler)', shape='box')
dot.node('M_Model', 'Model\n(XGBoost)', shape='box')
dot.node('M_SHAP', 'SHAP Explainer\n(shap)', shape='box')
dot.node('C_Pre', 'Preprocessing\n(Tokenization)', shape='box')
dot.node('C_Model', 'Model\n(Mistral/BERT)', shape='box')
dot.node('C_SHAP', 'SHAP Explainer\n(shap)', shape='box')
dot.node('Output', 'Output\n(Streamlit UI)')
dot.node('Feedback', 'Feedback Loop\n(Query Tuning)')

# Edges
dot.edge('Query', 'Task')
dot.edge('Task', 'Doping', label='Doping Intent')
dot.edge('Task', 'Match', label='Match-Fixing Intent')
dot.edge('Task', 'Cyber', label='Cyber Intent')
dot.edge('Doping', 'D_Pre')
dot.edge('D_Pre', 'D_Model')
dot.edge('D_Model', 'D_SHAP')
dot.edge('Match', 'M_Pre')
dot.edge('M_Pre', 'M_Model')
dot.edge('M_Model', 'M_SHAP')
dot.edge('Cyber', 'C_Pre')
dot.edge('C_Pre', 'C_Model')
dot.edge('C_Model', 'C_SHAP')
dot.edges([('D_SHAP', 'Output'), ('M_SHAP', 'Output'), ('C_SHAP', 'Output')])
dot.edge('Output', 'Feedback')
dot.edge('Feedback', 'Query', style='dashed')

# Render
dot.render('flow_diagram', view=True)