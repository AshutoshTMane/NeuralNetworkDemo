import streamlit as st


from components.information_button import floating_info_button  
from components.render_pages import render_page_1, render_page_2



# Main function
def main():
    
    # Select slider widget
    option = st.select_slider('Choose an option', options=['Off', 'On'])
    
    # Display different content based on the slider option
    if option == 'Off':
        render_page_1()
    else:
        render_page_2()


    #floating_info_button()
    




if __name__ == "__main__":
    main()
