import streamlit as st

def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Streamlit looks directly into .streamlit/secrets.toml
        if st.session_state["password"] == st.secrets["auth"]["ADMIN_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Clear password from memory
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Login UI
    st.title("🔒 Admin Login")
    st.text_input(
        "Please enter the administrator password", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
        
    return False