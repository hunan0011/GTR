import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW 
from tqdm import tqdm 

import config
from dataset import MsMarcoDataset, collate_fn, GetTargetPaths, MsMarcoDocVectorDataset, doc_vector_collate_fn
from model import Encoder, Indexer
from losses import InfoNCELoss, FocalLoss

# ================= Configuration =================
LAMBDA_DOC_AUX = getattr(config, 'LAMBDA_DOC_AUX', 1.0) 
LAMBDA_ROUTING = 1.0       
LAMBDA_CONTRAST = 1.0      
CONTRASTIVE_INTERVAL = 1 

def save_model(encoder, indexer, optimizer, scheduler, epoch, path):
    state = {
        'encoder': encoder.state_dict(),
        'indexer': indexer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 
        'epoch': epoch,
    }
    torch.save(state, path)
    tqdm.write(f"Checkpoint saved to {path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Datasets & Dataloaders
    print("Loading Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    dataset = MsMarcoDataset(
        queries_path=config.DOC_TRAIN_QUERIES, 
        qrels_path=config.DOC_TRAIN_QRELS,
        docid2path_path=config.ID2PATH,
        leaf2docs_path=config.LEAF2ID,
        doc_embedding_path=config.MEMMAP_PATH,          
        docid_to_index_path=config.ID2OFFSET,
        neg_num=config.NEG_NUM  
    )

    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=8, persistent_workers=True, prefetch_factor=2,
        collate_fn=lambda b: collate_fn(b, tokenizer, config.MAX_SEQ_LEN),
        pin_memory=True
    )

    doc_vec_dataset = MsMarcoDocVectorDataset(
        doc_embedding_path=config.MEMMAP_PATH,          
        docid_to_index_path=config.ID2OFFSET, 
        docid2path_path=config.ID2PATH,         
        length=len(dataset) 
    )
    
    doc_dataloader = DataLoader(
        doc_vec_dataset, batch_size=config.BATCH_SIZE*10, 
        shuffle=True, num_workers=4, collate_fn=doc_vector_collate_fn,
        pin_memory=True
    )
    doc_iterator = iter(doc_dataloader)

    # 2. Models & Optimization
    encoder = Encoder(model_name=config.MODEL_NAME, pooling=config.POOLING, device=device)
    indexer = Indexer(H=config.TREE_HEIGHT, B=config.NODE_BALANCE, device=device).to(device)
    path_getter = GetTargetPaths(config.ID2PATH, config.CHILDREN_EMBEDDINGS_PATH)

    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': config.LEARNING_RATE1},
        {'params': indexer.parameters(), 'lr': config.LEARNING_RATE2}
    ], weight_decay=config.WEIGHT_DECAY)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=30000, 
        num_training_steps=len(dataloader) * config.EPOCHS
    )

    focal_loss_fn = FocalLoss(gamma=2.0).to(device)
    contrastive_loss_fn = InfoNCELoss().to(device)

    # 3. Training Loop
    print("Starting Training...")
    
    for epoch in range(config.EPOCHS):
        encoder.train()
        indexer.train()
        
        do_contrastive = (epoch % CONTRASTIVE_INTERVAL == 0) 
        total_loss, total_route, total_cont, total_doc = 0.0, 0.0, 0.0, 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="step")
        
        for step, batch in enumerate(pbar):
            # Fetch Auxiliary Doc Batch
            try:
                doc_batch = next(doc_iterator)
            except StopIteration:
                doc_iterator = iter(doc_dataloader)
                doc_batch = next(doc_iterator)

            # Move Data to Device
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            d_emb_frozen = F.normalize(doc_batch["doc_emb"].to(device), p=2, dim=1)
            
            optimizer.zero_grad()

            # --- A. Query Path Forward ---
            q_emb = encoder(q_ids, q_mask)
            
            all_logits, all_targets = indexer(
                batch["qids"], q_emb, batch["pos_docids"], 
                batch["path_indices"], path_getter
            )
            
            loss_route = sum(focal_loss_fn(l, t) for l, t in zip(all_logits, all_targets))
            
            loss_cont = torch.tensor(0.0, device=device)
            if do_contrastive:
                pos_emb = F.normalize(batch["pos_emb"].to(device), p=2, dim=-1)
                neg_emb = F.normalize(batch["neg_emb"].to(device), p=2, dim=-1)
                scale_val = indexer.logit_scale.exp().clamp(max=100.0)
                loss_cont = contrastive_loss_fn(q_emb, pos_emb, neg_emb, logit_scale=scale_val)
            
            loss_main = (LAMBDA_ROUTING * loss_route) + (LAMBDA_CONTRAST * loss_cont)

            # --- B. Doc Aux Path Forward ---
            dummy_qids = torch.zeros(len(doc_batch["docids"]))
            doc_logits_list, doc_targets_list = indexer(
                dummy_qids, d_emb_frozen, doc_batch["docids"], 
                doc_batch["path_indices"], path_getter
            )
            
            loss_doc_aux = sum(focal_loss_fn(l, t) for l, t in zip(doc_logits_list, doc_targets_list))

            # --- C. Backward & Step ---
            loss = loss_main + (LAMBDA_DOC_AUX * loss_doc_aux)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(indexer.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            # --- Logging ---
            total_loss += loss.item()
            total_route += loss_route.item()
            total_cont += loss_cont.item()
            total_doc += loss_doc_aux.item()
            
            pbar.set_postfix(
                ls=f"{loss.item():.3f}", rt=f"{loss_route.item():.3f}", 
                ct=f"{loss_cont.item():.3f}", doc=f"{loss_doc_aux.item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}"
            )
            
        # End of Epoch
        steps = len(dataloader)
        print(f"Epoch {epoch+1} Avg: Loss={total_loss/steps:.4f} (Rt={total_route/steps:.4f}, Ct={total_cont/steps:.4f}, Doc={total_doc/steps:.4f})")

        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            save_path = os.path.join(config.OUTPUT_DIR, f"checkpoint-{epoch+1}.pt")
            save_model(encoder, indexer, optimizer, scheduler, epoch, save_path)
    
if __name__ == "__main__":
    main()

    