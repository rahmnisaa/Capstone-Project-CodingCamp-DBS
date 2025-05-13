import pandas as pd
import numpy as np
import csv

def generate_data(input_file_path):
    # Load CSV data
    with open(input_file_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)

    header = data[0]
    data_rows = data[1:]

    # Ubah ke DataFrame
    df = pd.DataFrame(data_rows, columns=header)

    # 2. product_id unik, diawali 1
    df['product_id'] = [int(f"1{i+1:04d}") for i in range(len(df))]

    # 3. store_id: ambil acak dari 1–2000, tidak semua nomor muncul
    store_ids = []
    store_id_mapping = {}
    i = 0

    # Ambil acak ID toko dari rentang 1–2000 (misal hanya 150 toko)
    unique_store_ids = np.random.choice(range(1, 2001), size=150, replace=False).astype(str)

    for store_id in unique_store_ids:
        n_products = np.random.randint(2, 51)  # 2–50 produk per toko
        store_id_mapping[store_id] = n_products
        for _ in range(n_products):
            if i >= len(df):
                break
            store_ids.append(store_id)
            i += 1

    # 4. customer_id: digit bebas sampai 4, tidak sama dengan store_id di baris yang sama
    n_customer_ids = 500
    customer_pool = set(str(i) for i in range(1, n_customer_ids + 1))

    # Tambahkan low-product store_id ke customer_pool
    low_product_store_ids = [sid for sid, count in store_id_mapping.items() if count < 5]
    customer_pool.update(low_product_store_ids)

    # Buat dictionary counter
    customer_id_counts = {cid: 0 for cid in customer_pool}
    customer_pool = list(customer_pool)  # convert to list for sampling

    assigned_customers = []

    # Store ID dengan <5 produk → harus muncul juga sebagai customer_id di baris berbeda
    low_product_store_ids = [sid for sid, count in store_id_mapping.items() if count < 5]
    low_store_customers_assigned = set()
    # Tambahkan store_id dulu
    df['store_id'] = store_ids
    for idx, row in df.iterrows():
        sid = row['store_id']
        available_customers = [cid for cid in customer_pool if cid != sid and customer_id_counts[cid] < 20]

        # Prioritaskan assign store_id kecil sebagai customer_id jika belum
        forced = None
        for low_sid in low_product_store_ids:
            if low_sid not in low_store_customers_assigned and low_sid != sid and customer_id_counts[low_sid] < 20:
                forced = low_sid
                break

        chosen = forced if forced else np.random.choice(available_customers)
        customer_id_counts[chosen] += 1
        if forced:
            low_store_customers_assigned.add(forced)
        assigned_customers.append(chosen)

    df['customer_id'] = assigned_customers

    # 5. Duplikasi data sampai total 2700 baris
    dupes_needed = 10000 - len(df)
    dup_rows = []

    while len(dup_rows) < dupes_needed:
        row = df.sample(1).iloc[0]
        n_dups = np.random.randint(2, 51)
        for _ in range(n_dups):
            if len(dup_rows) >= dupes_needed:
                break
            sid = row['store_id']
            available_customers = [cid for cid in customer_pool if cid != sid and customer_id_counts[cid] < 20]
            if not available_customers:
                break
            new_cust = np.random.choice(available_customers)
            customer_id_counts[new_cust] += 1

            dup_rows.append({
                'product_name': row['product_name'],
                'product_price_rp': row['product_price_rp'],
                'product_rating': row['product_rating'],
                'product_image_url': row['product_image_url'],
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'customer_id': new_cust
            })

    # 6. Gabungkan hasil akhir
    df_dupes = pd.DataFrame(dup_rows)
    df_final = pd.concat([df, df_dupes], ignore_index=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_final
