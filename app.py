
import streamlit as st
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from task_agent import handle_query

# Verify GROQ_API_KEY is loaded
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in .streamlit/secrets.toml. Ensure it exists with a valid key.")
    st.stop()

st.set_page_config(page_title="Agentic AI For Sports Violation Detection using Explainable AI", page_icon="🔍")

st.title("Agentic AI For Sports Violations Detection")
st.markdown("""
Enter a query to analyze for doping, match-fixing, or cyber violations.
The system will classify intents, extract data, impute missing fields, and provide agent results with explanations.
""")

# Query input
query = st.text_area("Enter your query:", 
                     placeholder="e.g., check hemoglobin:13.4, rbc=4.2; team_a_win_ratio=0.8; social post rbc=4.0",
                     height=150)

if st.button("Process Query"):
    if not query:
        st.error("Please enter a query.")
    else:
        try:
            logging.info(f"Processing query: {query}")
            with st.spinner("Processing query..."):
                results = handle_query(query)
                logging.info(f"Raw results from handle_query: {results}")
                
            if not results:
                st.warning("No valid intents detected.")
                logging.warning("No valid intents detected in results.")
            else:
                # Display results
                for result in results:
                    logging.info(f"Processing result: {result}")
                    intent = result.get("intent", "Unknown")
                    with st.expander(f"🔍 Intent: {intent.capitalize()}", expanded=True):
                        st.subheader("Sub-Query")
                        st.write(result.get("sub_query", "N/A"))
                        
                        st.subheader("Confidence")
                        confidence = result.get("confidence", "N/A")
                        try:
                            if isinstance(confidence, (int, float)):
                                st.write(f"{confidence:.2%}")
                            else:
                                st.write(str(confidence))
                                logging.warning(f"Confidence is not a number: {confidence}")
                        except Exception as e:
                            st.write("N/A")
                            logging.error(f"Error formatting confidence: {str(e)}")
                        
                        # st.subheader("Extracted Data")
                        # fields = result.get("result", {}).get("data", {})
                        # if fields:
                        #     st.json(fields)
                        # else:
                        #     st.write("No data extracted.")
                        #     logging.info("No extracted data for this intent.")
                        
                        # st.subheader("Imputed Fields")
                        # imputed = result.get("imputed_fields", [])
                        # if imputed:
                        #     st.write(", ".join(imputed))
                        # else:
                        #     st.write("No fields imputed.")
                        
                        # st.subheader("Missing Fields")
                        # missing = result.get("missing", [])
                        # if missing:
                        #     st.write(", ".join(missing))
                        # else:
                        #     st.write("All required fields provided.")
                        
                        st.subheader("Agent Result")
                        agent_result = result.get("result", {})
                        if "error" in agent_result:
                            st.error(f"Agent Error: {agent_result['error']}")
                            logging.error(f"Agent error for {intent}: {agent_result['error']}")
                        elif agent_result:
                            st.write(f"**Prediction**: {agent_result.get('prediction', 'N/A')}")
                            probability = agent_result.get('probability', 'N/A')
                            st.write(f"**Probability**: {probability}")
                            st.write(f"**Explanation**: {agent_result.get('explanation', 'No explanation provided.')}")
                            if intent == "cyber violation" and "predictions" in agent_result:
                                st.write("**Prediction Scores**:")
                                st.json(agent_result["predictions"])
                        else:
                            st.write("No agent result available.")
                            logging.warning(f"No agent result for {intent}.")
                        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logging.error(f"Error processing query: {str(e)}")
            st.write("Please check app.log for details.")

# Example query button
# if st.button("Try Example Query"):
#     example_query = (
#         "check if player is doping using hemoglobin:13.4, rbc=4.2, testosterone:5.9. "
#         "also check match fixing: team_a_win_ratio=0.8, team_b_win_ratio=0.4, red_cards=1."
#     )
#     st.session_state["query"] = example_query
#     query = example_query
#     logging.info("Running example query.")
#     st.rerun()

st.markdown("---")

















