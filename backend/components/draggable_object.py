import streamlit as st

def draggable_object():
    draggable_html = """
    <script>
        function createDraggableBox() {
            let existingBox = document.getElementById("draggable-box");
            if (existingBox) return; // Prevent multiple boxes from being added

            let box = document.createElement("div");
            box.id = "draggable-box";
            box.innerText = "Drag me";
            box.style.position = "fixed";
            box.style.width = "120px";
            box.style.height = "80px";
            box.style.backgroundColor = "lightblue";
            box.style.top = "50px";
            box.style.left = "50px";
            box.style.cursor = "grab";
            box.style.display = "flex";
            box.style.alignItems = "center";
            box.style.justifyContent = "center";
            box.style.fontWeight = "bold";
            box.style.borderRadius = "10px";
            box.style.zIndex = "99999";
            box.style.boxShadow = "2px 2px 10px rgba(0, 0, 0, 0.2)";
            document.body.appendChild(box);

            let offsetX, offsetY, isDragging = false;

            box.addEventListener("mousedown", function(e) {
                isDragging = true;
                offsetX = e.clientX - box.getBoundingClientRect().left;
                offsetY = e.clientY - box.getBoundingClientRect().top;
                box.style.cursor = "grabbing";
            });

            document.addEventListener("mousemove", function(e) {
                if (!isDragging) return;
                box.style.left = (e.clientX - offsetX) + "px";
                box.style.top = (e.clientY - offsetY) + "px";
            });

            document.addEventListener("mouseup", function() {
                isDragging = false;
                box.style.cursor = "grab";
            });
        }

        createDraggableBox();
    </script>
    """
    
    st.components.v1.html(draggable_html, height=300)

