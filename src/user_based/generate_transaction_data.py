import random
import pandas as pd

def generate_transaction_data(df, jumlah_customer=2500, jumlah_toko=50, total_transaksi=30000, max_usage=30, seed=42):
    random.seed(seed)
    
    # Salin data dan buat kolom ID dasar produk
    df = df.copy()
    df['product_base_id'] = ['3' + str(i).zfill(4) for i in range(1, len(df) + 1)]
    
    # Buat daftar customer dan pemilik toko
    customer_ids = list(range(1, jumlah_customer + 1))
    store_owner_ids = random.sample(customer_ids, jumlah_toko)
    
    # Distribusi produk ke toko
    store_products = {store_id: [] for store_id in store_owner_ids}
    all_store_product_rows = []
    product_counter = 1

    for _, row in df.iterrows():
        prob = random.random()
        if prob < 0.2:
            num_stores = random.randint(10, 20)
        elif prob < 0.7:
            num_stores = random.randint(2, 9)
        else:
            num_stores = 1

        selected_stores = random.sample(store_owner_ids, min(num_stores, len(store_owner_ids)))

        for store_id in selected_stores:
            product_id = '3' + str(product_counter).zfill(4)
            product_counter += 1

            product_row = {
                'product_id': product_id,
                'product_base_id': row['product_base_id'],
                'product_name': row['nama produk'],
                'product_image_url': row['gambar produk'],
                'product_price': row['harga produk'],
                'product_rating': row['rating produk'],
                'store_id': store_id
            }
            store_products[store_id].append(product_row)
            all_store_product_rows.append(product_row)
    
    # Generate transaksi
    transaksi_data = []
    product_usage = {row['product_id']: 0 for row in all_store_product_rows}

    for store_id, products in store_products.items():
        for produk in products:
            customer_id = random.choice([cid for cid in customer_ids if cid != store_id])
            transaksi_data.append({
                'transaction_id': len(transaksi_data) + 1,
                'customer_id': customer_id,
                'store_id': store_id,
                'product_id': produk['product_id'],
                'product_name': produk['product_name'],
                'product_image_url': produk['product_image_url'],
                'product_price': produk['product_price'],
                'product_rating': produk['product_rating']
            })
            product_usage[produk['product_id']] += 1

    while len(transaksi_data) < total_transaksi:
        store_id = random.choice(store_owner_ids)
        produk = random.choice(store_products[store_id])
        product_id = produk['product_id']

        if product_usage[product_id] < max_usage:
            customer_id = random.choice([cid for cid in customer_ids if cid != store_id])
            transaksi_data.append({
                'transaction_id': len(transaksi_data) + 1,
                'customer_id': customer_id,
                'store_id': store_id,
                'product_id': product_id,
                'product_name': produk['product_name'],
                'product_image_url': produk['product_image_url'],
                'product_price': produk['product_price'],
                'product_rating': produk['product_rating']
            })
            product_usage[product_id] += 1

    transaksi_df = pd.DataFrame(transaksi_data)
    return transaksi_df
