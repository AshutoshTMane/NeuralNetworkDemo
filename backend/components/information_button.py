import streamlit as st

def floating_info_button():
    """Creates a floating button that toggles a sidebar popup."""
    
    # Initialize session state for sidebar visibility
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = False

    # Floating button with HTML + CSS
    st.markdown(
        """
        <style>
            .floating-button {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 50px;
                height: 50px;
                background-color: #ff4b4b;
                color: white;
                border-radius: 50%;
                text-align: center;
                font-size: 24px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
                cursor: pointer;
                line-height: 50px;
                user-select: none;
                transition: 0.3s;
            }
            .floating-button:hover {
                background-color: #cc3b3b;
            }
        </style>
        <div class="floating-button" onclick="toggleSidebar()">+</div>
        
        <script>
            function toggleSidebar() {
                window.parent.postMessage({type: 'streamlit', key: 'toggle_sidebar'}, '*');
            }
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Capture JavaScript message to toggle sidebar in session state
    if st.query_params.get("toggle_sidebar") == "true":
        st.session_state.sidebar_open = not st.session_state.sidebar_open

    # Show sidebar when activated
    if st.session_state.sidebar_open:
        with st.sidebar:
            st.write("This is a sidebar popup!")
            if st.button("Close Sidebar"):
                st.session_state.sidebar_open = False
