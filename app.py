from flask import Flask, request, render_template, send_file
from gradio_client import Client, handle_file
from PIL import Image, ImageOps
from io import BytesIO
import requests
import cloudinary
import cloudinary.uploader
import os
import base64



app = Flask(__name__)





# Cloudinary and remove.bg API setup
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY", "p1SChAJPV2sjC4PqWjtQVJ8w")
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dcajb02df"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "862192414383365"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "TDuIQPd_iRf5_ThniMlwn8Gaaq8")
)


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print("==== /process endpoint hit ====")
   
    if 'image' not in request.files:
        print("DEBUG: No image in request")
        return "No image uploaded", 400
    
    
    file = request.files['image']
    print(f"DEBUG: Received image file: {file.filename}")
    input_image = file.read()

    # Layout settings
    passport_width = 384
    passport_height = 472
    border = 2
    spacing = 25
    margin_x = 10
    margin_y = 15
    horizontal_gap = 20
    a4_w, a4_h = 2480, 3508
    copies = int(request.form.get("copies", 6))
    print(f"DEBUG: Copies requested = {copies}")

    # Step 1: Background removal
    print("DEBUG: Sending image to remove.bg...")
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': input_image},
        data={'size': 'auto'},
        headers={'X-Api-Key': REMOVE_BG_API_KEY}
    )
    print(f"DEBUG: remove.bg response status = {response.status_code}")
    if response.status_code != 200:
     print(f"ERROR: Background removal failed - {response.text}")
     try:
        error_info = response.json()
        if error_info.get("errors"):
            error_code = error_info["errors"][0].get("code", "unknown_error")
            return {"error": error_code}, 410
     except Exception as ex:
        print("Failed to parse error details:", ex)

     return {"error": "bg_removal_failed"}, 500


    bg_removed = BytesIO(response.content)
    img = Image.open(bg_removed)
    print(f"DEBUG: Image mode after background removal: {img.mode}")

    if img.mode in ("RGBA", "LA"):
        print("DEBUG: Converting transparent background to white")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        processed_img = background
    else:
        processed_img = img.convert("RGB")

    # Step 3: Upload to Cloudinary
    buffer = BytesIO()
    processed_img.save(buffer, format="PNG")
    buffer.seek(0)
    print("DEBUG: Uploading to Cloudinary...")
    upload_result = cloudinary.uploader.upload(buffer, resource_type="image")
    image_url = upload_result.get("secure_url")
    print(f"DEBUG: Cloudinary URL: {image_url}")
    if not image_url:
        print("ERROR: Failed to get image URL from Cloudinary.")
        return "Cloudinary upload failed", 500

    # Step 4: Upscale via Hugging Face
    print("DEBUG: Downloading image from Cloudinary for enhancement...")

    image_dict = handle_file(image_url)

    gallery = [{"image": image_dict, "caption": None}]

    client = Client("naman14113114/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN")

    result = client.predict(
        gallery=gallery,
        face_restoration="GFPGANv1.4.pth",
        upscale_model="SRVGG, realesr-general-x4v3.pth",
        scale=2,
        face_detection="retinaface_resnet50",
        face_detection_threshold=10,
        face_detection_only_center=False,
        outputWithModelName=True,
        api_name="/inference"
    )

    last_image_path = result[1][-1]


    
    img = Image.open(last_image_path)

    

    # Step 5: RGB conversion
    if img.mode in ("RGBA", "LA"):
        print("DEBUG: Replacing transparency with white again post-enhancement")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        passport_img = background
    else:
        passport_img = img.convert("RGB")

    # Step 6: Resize and border
    passport_img = passport_img.resize((passport_width, passport_height), Image.LANCZOS)

    passport_img = ImageOps.expand(passport_img, border=border, fill='black')
    print(f"DEBUG: Passport image size after border = {passport_img.size}")

    # Step 7: Compose A4 layout
    a4 = Image.new("RGB", (a4_w, a4_h), "white")
    x, y = margin_x, margin_y
    paste_w = passport_width + 2 * border
    paste_h = passport_height + 2 * border
    placed = 0

    print("DEBUG: Placing images onto A4 sheet...")
    for _ in range(copies):
        if x + paste_w > a4_w:
            x = margin_x
            y += paste_h + spacing
        if y + paste_h > a4_h:
            print("DEBUG: Reached end of page")
            break
        a4.paste(passport_img, (x, y))
        print(f"DEBUG: Placed copy {placed + 1} at x={x}, y={y}")
        x += paste_w + horizontal_gap
        placed += 1

    print(f"DEBUG: Total placed = {placed}")

    # Step 8: Export to PDF
    output = BytesIO()
    a4.save(output, format="PDF", dpi=(300, 300))
    output.seek(0)
    print("DEBUG: Returning PDF file to client.")

    return send_file(output, mimetype="application/pdf", as_attachment=True, download_name="passport-sheet.pdf")


if __name__ == '__main__':
    app.run(debug=True)
