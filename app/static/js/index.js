// static/js/index.js

// この画像だけデータセットZIPエクスポート
window.exportCurrentImageDataset = async function() {
    if (!images || images.length === 0) {
        alert('画像がありません');
        return;
    }

    const image = images[currentImageIndex];
    if (!image || !image.id) {
        alert('画像情報が取得できません');
        return;
    }

    const btn = document.querySelector('.export-button[onclick="exportCurrentImageDataset()"]');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'エクスポート中...';
    }

    try {
        const res = await fetch(`/api/export_single/${image.id}`);
        if (!res.ok) throw new Error('エクスポートに失敗しました');
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${image.filename.replace(/\.[^.]+$/, '')}_dataset.zip`;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }, 100);
    } catch (e) {
        alert('エクスポートに失敗しました');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'この画像だけデータセットZIP';
        }
    }
};
