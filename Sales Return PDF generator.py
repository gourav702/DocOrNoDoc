import os
import random
from fpdf import FPDF
from datetime import datetime, timedelta

# Create directory for the files
output_dir = "Generated_Sales_Returns"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data pools to ensure every document is unique
vendors = [
    "Sehgal Consultancy", "Global Tech Solutions", "Apex Retailers",
    "Vibrant Enterprises", "Drona Logistics", "Hindustan Traders",
    "Nordic Systems", "Cloud Nine Services", "Elite Machineries", "Sunrise Corp"
]
parties = [
    "Heels & Craft", "Urban Styles", "Metro Footwear", "Blue Chip Inc",
    "Rapid Delivery", "Classic Furnitures", "Modern Gadgets", "Style Quotient"
]
items = [
    "Key Board+Mouse Combo", "Laser Printer Toner", "1TB External HDD",
    "Ergonomic Office Chair", "USB-C Hub", "27-inch Monitor",
    "Wireless Router", "Bluetooth Headphones", "Webcam 1080p"
]
cities = ["Delhi", "Jaipur", "Mumbai", "Bangalore", "Pune", "Ahmedabad"]


def generate_random_gstin():
    return f"{random.randint(10, 99)}AAAPS{random.randint(1000, 9999)}P1Z{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"


def create_pdf(index):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    vendor = random.choice(vendors)
    party = random.choice(parties)

    # Header Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"GSTIN: {generate_random_gstin()}", ln=True, align='L')
    pdf.cell(200, 10, txt="Credit Note", ln=True, align='C')  # Critical keyword for classification

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=vendor, ln=True, align='R')

    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 5,
                   f"Address: {random.randint(1, 999)}, Industrial Area, {random.choice(cities)}\nTel: {random.randint(9000000000, 9999999999)}\nEmail: contact@{vendor.lower().replace(' ', '')}.com",
                   align='R')

    pdf.ln(5)

    # Party Details
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 10, txt="Party Details :", ln=False)
    pdf.cell(100, 10, txt=f"Cr. Note No: CN-{random.randint(1000, 9999)}", ln=True, align='R')

    pdf.set_font("Arial", size=10)
    pdf.cell(100, 7, txt=party, ln=False)
    pdf.cell(100, 7, txt=f"Date: {(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%d-%m-%Y')}",
             ln=True, align='R')

    pdf.multi_cell(100, 5,
                   f"{random.randint(100, 500)}, Market Street, {random.choice(cities)}\nGSTIN / UIN : {generate_random_gstin()}")

    pdf.ln(10)

    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(10, 10, "S.N.", 1)
    pdf.cell(80, 10, "Description of Goods", 1)
    pdf.cell(20, 10, "HSN", 1)
    pdf.cell(20, 10, "Qty", 1)
    pdf.cell(30, 10, "Price", 1)
    pdf.cell(30, 10, "Amount", 1)
    pdf.ln()

    # Table Content
    pdf.set_font("Arial", size=10)
    qty = random.randint(1, 10)
    price = random.choice([100, 250, 500, 1200, 5000])
    total = qty * price

    pdf.cell(10, 10, "1.", 1)
    pdf.cell(80, 10, random.choice(items), 1)
    pdf.cell(20, 10, str(random.randint(1000, 9999)), 1)
    pdf.cell(20, 10, f"{qty}.000", 1)
    pdf.cell(30, 10, f"{price:,.2f}", 1)
    pdf.cell(30, 10, f"{total:,.2f}", 1)
    pdf.ln()

    # Totals
    pdf.cell(130, 10, "", 0)
    pdf.cell(30, 10, "Grand Total", 1)
    pdf.cell(30, 10, f"{total:,.2f}", 1)

    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "This is a computer generated Credit Note.", ln=True, align='C')
    pdf.cell(0, 10, f"For {vendor} - Authorised Signatory", ln=True, align='R')

    file_name = f"SalesReturn_{index + 1}.pdf"
    pdf.output(os.path.join(output_dir, file_name))
    print(f"Generated: {file_name}")


# Generate 30 files
for i in range(30):
    create_pdf(i)

print("\n✅ All 30 Sales Returns generated in the 'Generated_Sales_Returns' folder.")