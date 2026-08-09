/* ══════════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════════ */
let HOST = 'http://triton-api:8003';
let allModels    = [];   // single / ONNX models
let allEnsembles = [];   // ensemble models
let currentFile = null;
let currentImage = null;
let uploadModelFile = null;
let ensSteps = [];
let editingEnsemble = null; // null = create, string = tên ensemble đang edit
let addStreamTab = 'stream';
let nsSelectedFile = null;
let streamIdCounter = 0;
let detectMode = 'image';
let detectWebcamStream = null;
let detectWebcamVideo = null;
let detectWebcamLoopId = null;
let detectWebcamWs = null;
let detectWebcamAnns = [];
let detectWebcamImgShape = null;
let detectWebcamCatMap = {};
let detectWebcamGeneration = 0;
let detectWebcamExactFrame = null;
let detectWebcamExactFrameAt = 0;
let videoDetectWs = null;
let videoDetectLoopId = null;
let videoDetectAnns = [];
let videoDetectImgShape = null;
let videoDetectCatMap = {};
let videoDetectExactFrame = null;
let videoDetectExactFrameAt = 0;
let modelInfoCache = {};
let apiMaxFps = 30;
let gpuInfoCache = { gpus: [], models_by_gpu: {}, default_gpu: null };
let apiHealthCache = null;
let currentLanguage = 'en';
let detectLivePerf = { mode: '', windowStart: 0, frames: 0, measuredFps: 0 };

// Trajectory Trails removed

const I18N = {
  en: {},
  vi: {
    'nav.detect': 'Phát Hiện',
    'nav.stream': 'Xem Camera',
    'nav.tracking': 'Theo Dõi',
    'nav.system': 'Cài Đặt',
    'nav.keys': 'Khóa API',
    'nav.logout': 'Đăng Xuất',
    'action.connect': 'Kết Nối',
    'panel.image_inference': 'Suy Luận Hình Ảnh',
    'mode.image': 'Ảnh',
    'mode.video': 'Video',
    'mode.webcam': 'Webcam',
    'drop.image': 'Thả ảnh vào đây hoặc bấm để chọn',
    'drop.video': 'Thả video vào đây hoặc bấm để chọn',
    'panel.parameters': 'Thông Số',
    'panel.detections': 'Phát Hiện',
    'panel.loaded_models': 'Mô Hình Đã Tải',
    'panel.upload_model': 'Tải Mô Hình Lên',
    'panel.model_config': 'Cấu Hình Mô Hình & Nhãn',
    'panel.active_ensembles': 'Tổ Hợp Đang Hoạt Động',
    'panel.create_ensemble': 'Tạo Tổ Hợp',
    'panel.system_health': 'Sức Khỏe Hệ Thống',
    'panel.gpu_info': 'Thông Tin GPU',
    'panel.system_monitor': 'Giám Sát Hệ Thống',
    'panel.triton_stats': 'Thống Kê Suy Luận Triton',
    'action.models': 'Mô Hình',
    'action.add_stream': 'Thêm Luồng',
    'action.stop_all': 'Dừng Tất Cả',
    'action.refresh': 'Làm Mới',
    'action.reload': 'Tải Lại',
    'action.open': 'Mở',
    'action.run_inference': 'Chạy Suy Luận',
    'action.start_detect': 'Bắt Đầu Phát Hiện',
    'action.start_webcam': 'Bắt Đầu Webcam',
    'action.upload_export': 'Tải Lên & Xuất',
    'action.cancel_edit': '✕ Hủy chỉnh sửa',
    'option.live_preview': 'Khung Chính Xác - Trễ Tốt',
    'option.exact_latency': 'Khung Chính Xác - Trễ Tốt',
    'option.native_exact': 'FPS Gốc - Vẽ JSON',
    'option.exact_boxes': 'Khung Chính Xác - Bám Tốt Nhất',
    'option.synced_boxes': 'Khung Chính Xác - Trễ Tốt',
    'option.tracked_live': 'Bám Theo Live - Khuyến Nghị',
    'empty.no_streams': 'Chưa có luồng hoạt động',
    'empty.no_results': 'Chưa có kết quả',
    'empty.no_models': 'Chưa tải mô hình',
    'Fullscreen': 'Toàn Màn Hình',
    'Delete': 'Xóa',
    'Delete stream': 'Xóa luồng',
    'Delete model': 'Xóa mô hình',
    'Delete ensemble': 'Xóa tổ hợp',
    'Revoke': 'Thu hồi',
    'Create Key': 'Tạo Khóa',
    'Create Secret Key': 'Tạo Khóa Bí Mật',
    'Key Name': 'Tên Khóa',
    'Model Access': 'Quyền Truy Cập Mô Hình',
    'All Loaded Models': 'Tất Cả Mô Hình',
    'Restricted Models': 'Mô Hình Giới Hạn',
    'Expiration': 'Hạn Dùng',
    'Never (Perpetual)': 'Không Hết Hạn',
    'Select models...': 'Chọn mô hình...',
    'models selected': 'mô hình được chọn',
    'Stream 1': 'Luồng 1',
    'Add Stream': 'Thêm Luồng',
    'Detect': 'Phát Hiện',
    'Image': 'Hình Ảnh',
    'Video': 'Video',
    'Webcam': 'Webcam',
    'Run Inference': 'Chạy Suy Luận',
    'Stop': 'Dừng',
    'Edit': 'Sửa',
    'Apply': 'Áp Dụng',
    'Show': 'Hiện',
    'Hide': 'Ẩn',
    'Close': 'Đóng',
    'Refresh': 'Làm Mới',
    'Save Changes': 'Lưu Thay Đổi',
    'Cancel': 'Hủy',
    'tracking.similarity_threshold': 'Ngưỡng Tương Đồng:',
    'tracking.threshold_broad': '50% (Mở Rộng)',
    'tracking.threshold_rec': '70% (Khuyên Dùng)',
    'tracking.threshold_precise': '95% (Chính Xác Cao)',
  },
};

const VI_LABELS = {
  Performance: 'Hiệu Năng',
  Parameters: 'Thông Số',
  Detections: 'Phát Hiện',
  time: 'Thời Gian',
  fps: 'FPS',
  model: 'Mô Hình',
  total: 'Tổng',
  server: 'Máy Chủ',
  triton: 'Triton',
  post: 'Hậu Xử Lý',
  'preview in': 'Tải Preview',
  'infer out': 'Gửi Suy Luận',
  'json in': 'Nhận JSON',
  reconn: 'Kết Nối Lại',
  'detect fps': 'FPS Phát Hiện',
  'infer cap': 'Giới Hạn Suy Luận',
  'preview fps': 'FPS Preview',
  'preview cap': 'Giới Hạn Preview',
  imgsz: 'Kích Thước Ảnh',
  shape: 'Khung Hình',
  models: 'Mô Hình',
  conf: 'Độ Tin Cậy',
  classes: 'Lớp',
  sync: 'Đồng Bộ',
  watch: 'Theo Dõi',
  stall: 'Treo',
  labels: 'Nhãn',
  gpus: 'GPU',
  'live fps': 'FPS Trực Tiếp',
  'infer fps': 'FPS Suy Luận',
  preview: 'Preview',
  'server preview': 'Preview Máy Chủ',
  'server sync': 'Đồng Bộ Máy Chủ',
  'server overlay on': 'Vẽ Trên Máy Chủ',
  'box sync': 'Đồng Bộ Khung',
  overlay: 'Hiển Thị',
  'Live Preview': 'Khung Chính Xác',
  'Exact Boxes': 'Khung Chính Xác',
  'Native FPS Exact': 'FPS Gốc - Vẽ JSON',
  'Native FPS Live': 'FPS Gốc - Vẽ JSON',
  'Use exact inference frame for boxes and masks': 'Dùng đúng khung suy luận để vẽ khung và mask',
  'auto reconnect': 'Tự Kết Nối Lại',
  'Reconnect Source': 'Kết Nối Lại Nguồn',
  Apply: 'Áp dụng',
  Show: 'Hiện',
  Hide: 'Ẩn',
  Close: 'Đóng',
  Edit: 'Sửa',
  'Audio Off': 'Tắt Âm Thanh',
  'Audio On': 'Bật Âm Thanh',
  Stop: 'Dừng',
  Waiting: 'Đang Chờ',
  'No detections': 'Không Có Phát Hiện',
  iou: 'IoU',
  
  // Additional translations
  'tracking': 'Theo dõi',
  'All': 'Tất Cả',
  'None': 'Không',
  'GPU': 'GPU',
  'Name': 'Tên',
  'VRAM': 'VRAM',
  'Models': 'Mô Hình',
  'System Health': 'Sức Khỏe Hệ Thống',
  'Models Config': 'Cấu Hình Mô Hình',
  'Ensembles': 'Tổ Hợp Mô Hình',
  'API Keys': 'Khóa API',
  'Secret Key': 'Khóa Bí Mật',
  'Created By': 'Người Tạo',
  'Scopes': 'Phạm Vi',
  'Allowed Models': 'Mô Hình Được Phép',
  'Created': 'Ngày Tạo',
  'Expires': 'Ngày Hết Hạn',
  'Action': 'Hành Động',
  'Object Tracking': 'Theo Dõi Đối Tượng',
  'Tracked Objects Gallery': 'Thư Viện Đối Tượng',
  'All classes': 'Tất Cả Các Lớp',
  'Refresh': 'Làm Mới',
  'Clear All': 'Xóa Tất Cả',
  'Select All': 'Chọn Tất Cả',
  'Deselect All': 'Bỏ Chọn Tất Cả',
  'Delete Selected': 'Xóa Đã Chọn',
  'Enable ByteTrack Tracking': 'Kích Hoạt Theo Dõi ByteTrack',
  'Detection FPS': 'FPS Phát Hiện',
  'Src Bandwidth': 'Băng Thông Nguồn',
  'Infer Upload': 'Tải Lên Suy Luận',
  'JSON Bandwidth': 'Băng Thông JSON',
  'Avg Latency': 'Độ Trễ Trung Bình',
  'Max Latency': 'Độ Trễ Tối Đa',
  '+ Add RTSP Camera': '+ Thêm Camera RTSP',
  'Active Streams': 'Luồng Đang Hoạt Động',
  'Active Ensembles': 'Tổ Hợp Hoạt Động',
  'Upload Model (.pt / .onnx)': 'Tải Lên Mô Hình (.pt / .onnx)',
  'Create Ensemble': 'Tạo Tổ Hợp',
  'GPU Info': 'Thông Tin GPU',
  'System Monitor': 'Giám Sát Hệ Thống',
  'Triton Inference Stats': 'Thống Kê Triton',
  'No recorded clips found for this stream.': 'Không tìm thấy clip nào cho luồng này.',
  'No active API keys found. Click "+ Create new secret key" to generate one.': 'Chưa có khóa API nào. Bấm "+ Tạo khóa bí mật mới" để bắt đầu.',
  'No tracked objects yet.': 'Chưa có đối tượng nào được theo dõi.',
  'Start a stream with a detection model to populate.': 'Chạy camera với một mô hình để hiển thị ở đây.',
  'Connect to load stats': 'Kết nối để tải số liệu',
  'No stats loaded': 'Chưa tải số liệu',
  '+ Create new secret key': '+ Tạo khóa bí mật mới',
  'Manage secret keys used to authenticate requests to the Triton Inference Server. Do not expose keys in browser client side scripts.': 'Quản lý các khóa bí mật dùng để xác thực các yêu cầu gửi đến Triton Inference Server. Không để lộ các khóa này trong các mã chạy phía trình duyệt.',
  'Triton Admin Portal': 'Cổng Quản Trị Triton',
  'Enter password to access server streams & key management.': 'Nhập mật khẩu để quản lý camera và khóa.',
  'Account Name': 'Tên Tài Khoản',
  'Password': 'Mật Khẩu',
  'Remember me': 'Duy trì đăng nhập',
  'Log In': 'Đăng Nhập',
  Accounts: 'Tài Khoản',
  'Manage Accounts': 'Quản Lý Tài Khoản',
  'Create New Account': 'Tạo Tài Khoản Mới',
  'Username': 'Tên Tài Khoản',
  'Role': 'Vai Trò',
  'Actions': 'Thao Tác',
  'Create Account': 'Tạo Tài Khoản',
  'Save Changes': 'Lưu Thay Đổi',
  'Cancel': 'Hủy'
};

const VI_TITLES = {
  'Latency and bandwidth counters. Hidden by default so the config panel stays stable.': 'Bộ đếm độ trễ và băng thông. Mặc định ẩn để khung cấu hình ổn định.',
  'Browser round-trip latency for the latest inference result.': 'Độ trễ vòng đi-về của trình duyệt cho kết quả suy luận mới nhất.',
  'Detection results received per second.': 'Số kết quả phát hiện JSON nhận được mỗi giây.',
  'Download from API to browser. In RTSP Camera mode this is the optional server-side preview stream.': 'Băng thông tải từ API về trình duyệt. Với RTSP Camera, đây là luồng preview từ máy chủ.',
  'Upload from browser to API inference WebSocket. RTSP Camera server-side mode should be zero because API reads and infers directly.': 'Băng thông gửi từ trình duyệt tới WebSocket suy luận của API. Với RTSP Camera phía máy chủ, giá trị này gần như bằng 0 vì API đọc và suy luận trực tiếp.',
  'Detection JSON download from API to browser.': 'Băng thông tải JSON kết quả phát hiện từ API về trình duyệt.',
  'Changing these values reconnects inference with the new settings.': 'Đổi các giá trị này sẽ kết nối lại suy luận với cấu hình mới.',
  'Models run against this stream. More models means more WebSockets and more GPU/API load.': 'Các mô hình chạy trên luồng này. Nhiều mô hình hơn sẽ tăng tải GPU/API.',
  'Minimum confidence score. Higher removes weak detections; lower shows more boxes.': 'Điểm tin cậy tối thiểu. Cao hơn sẽ bỏ phát hiện yếu; thấp hơn sẽ hiện nhiều khung hơn.',
  'Input size sent to the model. Larger can improve small objects but costs more latency.': 'Kích thước đầu vào gửi vào mô hình. Lớn hơn có thể tốt hơn cho vật nhỏ nhưng tăng độ trễ.',
  'Inference FPS cap. For RTSP Camera mode, API samples this many frames per second from the camera for detection.': 'Giới hạn FPS suy luận. Với RTSP Camera, API lấy tối đa số khung hình này mỗi giây để phát hiện.',
  'RTSP preview FPS. Lower saves browser download bandwidth.': 'FPS preview RTSP. Giảm giá trị này để tiết kiệm băng thông tải về trình duyệt.',
  'Preview FPS sent from API back to this browser. Lower saves network bandwidth without reducing detection FPS.': 'FPS preview API gửi về trình duyệt. Giảm giá trị này giúp tiết kiệm băng thông mà không giảm FPS phát hiện.',
  'YOLOE text prompts only. Normal YOLO models use labels.json.': 'Chỉ dành cho prompt văn bản YOLOE. YOLO thường dùng labels.json.',
  'Managed RTSP draws boxes on the API server before sending the preview frame.': 'RTSP quản lý vẽ khung trên API server trước khi gửi ảnh preview.',
  'annotated preview': 'preview có khung',
  'API draws boxes': 'API vẽ khung',
  'When on, API draws boxes on preview JPEG. When off, browser draws boxes from JSON events (like webcam streams).': 'Bật: API vẽ khung lên JPEG preview. Tắt: trình duyệt vẽ khung từ JSON events (giống webcam).',
  'When on, API draws boxes on the exact frame it sent to inference. Best alignment for fast objects, but the preview can feel delayed.': 'Bật: API vẽ khung trên đúng khung hình đã suy luận. Khớp tốt nhất với vật chạy nhanh nhưng preview có thể trễ hơn.',
  'Live Preview is lowest visual latency. Exact Boxes draws results on the frame sent to inference, so fast objects align better but the visible annotated frame can be delayed.': 'Khung Chính Xác vẽ kết quả trên đúng khung hình đã suy luận, giúp box/mask bám vật thể nhanh tốt hơn.',
  'Exact-frame mode draws boxes/masks on the same frame used for inference.': 'Chế độ khung chính xác vẽ box/mask trên đúng khung hình đã suy luận.',
  'Tracker used for object IDs. ByteTrack uses Kalman filter motion matching with 2-stage IoU association.': 'Bộ bám dùng để giữ ID vật thể. ByteTrack sử dụng bộ lọc Kalman và ghép nối IoU 2 giai đoạn.',
  'Exact Boxes draws boxes/masks on the same frame used for inference. Best alignment for fast objects; visual preview can be delayed by inference latency.': 'Khung Chính Xác vẽ box/mask trên cùng khung hình dùng để suy luận. Bám vật nhanh tốt nhất, nhưng preview có thể trễ theo thời gian suy luận.',
  'Exact Boxes draws boxes/masks on the same frame sent to inference. Better for fast vehicles/people; off uses the newest live frame.': 'Khung Chính Xác vẽ box/mask trên đúng khung gửi đi suy luận. Tốt hơn cho xe/người di chuyển nhanh; tắt để dùng khung live mới nhất.',
  'Native FPS Live shows the newest source frame and draws the newest detection JSON in the browser. It is smoother, but boxes can lag by inference latency.': 'FPS Gốc hiển thị khung mới nhất từ nguồn và vẽ JSON phát hiện mới nhất trên trình duyệt. Mượt hơn, nhưng box có thể trễ theo thời gian suy luận.',
  'Draw boxes on the exact frame sent to inference. Better for fast vehicles/people; off uses the latest live frame for a smoother but less exact overlay.': 'Vẽ khung trên đúng khung hình đã gửi đi suy luận. Tốt hơn cho xe/người chạy nhanh; tắt để preview mượt hơn nhưng khung có thể lệch.',
  'Use exact inference frame for boxes': 'Dùng đúng khung suy luận để vẽ box',
  'server draws boxes': 'máy chủ vẽ khung',
  'client draws boxes': 'trình duyệt vẽ khung',
  'Reconnect the source WebSocket if frames stop arriving.': 'Tự kết nối lại nguồn nếu không còn nhận khung hình.',
  'Seconds without source frames before auto reconnect triggers.': 'Số giây không có khung hình trước khi tự kết nối lại.',
  'For legacy RTSP/WebSocket proxy, draw boxes on the exact frame sent to inference. Better alignment for fast objects, but preview can feel slightly less live.': 'Với bridge RTSP/WebSocket cũ, vẽ khung trên đúng khung hình đã gửi đi suy luận. Khớp tốt hơn với vật di chuyển nhanh nhưng preview có thể kém trực tiếp hơn.',
  'Draw boxes on the exact frame sent to inference. Better alignment for fast objects, but the displayed annotated frame can feel delayed.': 'Vẽ khung trên đúng khung hình đã gửi đi suy luận. Khớp tốt hơn với vật di chuyển nhanh, nhưng khung hình có chú thích có thể trễ hơn.',
  'Custom FPS cap. Clamped to API max_fps.': 'Giới hạn FPS tùy chỉnh. Bị giới hạn bởi max_fps của API.',
  'Reconnect source if no RTSP/WS frame arrives for this many seconds': 'Kết nối lại nguồn nếu không có khung RTSP/WS trong số giây này.',
  'Download bandwidth for preview frames. RTSP Camera mode receives these from /streams/{id}/preview; legacy RTSP bridge receives them from /ws/rtsp.': 'Băng thông tải khung preview. RTSP Camera nhận từ /streams/{id}/preview; bridge RTSP cũ nhận từ /ws/rtsp.',
  'Upload bandwidth from this browser to API /ws/stream for inference frames. RTSP Camera server-side mode does not upload frames, so this should stay near zero.': 'Băng thông trình duyệt gửi khung suy luận tới API /ws/stream. RTSP Camera phía máy chủ không gửi khung từ trình duyệt nên gần như bằng 0.',
  'Download bandwidth for detection JSON results from API to this browser.': 'Băng thông tải JSON phát hiện từ API về trình duyệt.',
  'How many times the RTSP preview/source WebSocket reconnected.': 'Số lần WebSocket preview/nguồn RTSP đã kết nối lại.',
  'Maximum inference frames per second requested by the client, clamped by API max_fps.': 'FPS suy luận tối đa client yêu cầu, bị giới hạn bởi max_fps của API.',
  'Measured preview JPEG frames per second received by the browser.': 'Số khung JPEG preview thực tế trình duyệt nhận mỗi giây.',
  'Configured preview JPEG FPS sent from API to browser.': 'FPS JPEG preview đã cấu hình để API gửi về trình duyệt.',
  'Inference input size after letterbox resize.': 'Kích thước đầu vào suy luận sau khi resize letterbox.',
  'Original frame height and width used to map boxes back.': 'Chiều cao/rộng khung gốc dùng để map khung phát hiện về đúng vị trí.',
  'Open stream settings': 'Mở Cài Đặt Luồng',
  'Close stream settings': 'Đóng Cài Đặt Luồng',
  'Detect': 'Phát hiện',
  'Stream': 'Xem camera',
  'Tracking': 'Theo dõi',
  'Settings': 'Cài đặt',
  'API Keys': 'Khóa API',
  'Logout': 'Đăng xuất',
  'Toggle Theme': 'Giao diện Sáng/Tối',
};

function tr(key, fallback = '') {
  return I18N[currentLanguage]?.[key] || fallback || key;
}

function uiLabel(label) {
  return currentLanguage === 'vi' ? (VI_LABELS[label] || label) : label;
}

function uiTitle(text) {
  return currentLanguage === 'vi' ? (VI_TITLES[text] || text) : text;
}

function setLanguage(lang, userTriggered = false) {
  const previousLang = currentLanguage;
  currentLanguage = lang === 'vi' ? 'vi' : 'en';
  localStorage.setItem('admin_lang', currentLanguage);
  document.documentElement.lang = currentLanguage;
  
  const sel = document.getElementById('lang-select');
  if (sel) sel.value = currentLanguage;

  if (userTriggered && previousLang !== currentLanguage) {
    window.location.reload();
    return;
  }

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const fallback = el.getAttribute('data-i18n-default') || el.textContent;
    if (!el.getAttribute('data-i18n-default')) el.setAttribute('data-i18n-default', fallback);
    el.textContent = tr(key, fallback);
  });
  document.querySelectorAll('[title]').forEach(el => {
    const fallback = el.getAttribute('data-title-default') || el.getAttribute('title') || '';
    if (!el.getAttribute('data-title-default')) el.setAttribute('data-title-default', fallback);
    el.setAttribute('title', currentLanguage === 'vi' ? (VI_TITLES[fallback] || fallback) : fallback);
  });
  document.querySelectorAll('[data-dyn-label]').forEach(el => {
    const label = el.getAttribute('data-dyn-label') || el.textContent;
    el.textContent = currentLanguage === 'vi' ? (VI_LABELS[label] || label) : label;
  });

  // Auto-translate common text selectors to catch untranslated elements
  const selectors = [
    '.param-label', 
    '.panel-title', 
    'th.admin-keys-th', 
    'button.px-3.py-2.text-xs', 
    '.section-title',
    'th',
    'button.btn',
    '.sidebar-btn-text',
    '.stream-toolbar-label',
    '.admin-keys-title',
    '.admin-keys-subtitle',
    '.empty-state',
    '.modal-backdrop h3',
    '.modal-backdrop label',
    '.modal-backdrop div',
    '.modal-backdrop span',
    '.modal-backdrop p',
    '.modal-backdrop option',
    '.modal-backdrop button',
    '#trk-clips-title'
  ];
  document.querySelectorAll(selectors.join(',')).forEach(el => {
    if (el.getAttribute('data-i18n') || el.getAttribute('data-dyn-label')) return;
    
    // Guard: Only translate leaf nodes (length of children is 0) to avoid stripping format tags or icons
    if (el.children && el.children.length > 0) return;
    
    const text = el.textContent.trim();
    if (!text) return;
    if (VI_LABELS[text]) {
      const fallback = el.getAttribute('data-dyn-label-default') || el.textContent;
      if (!el.getAttribute('data-dyn-label-default')) {
        el.setAttribute('data-dyn-label-default', fallback);
      }
      el.textContent = currentLanguage === 'vi' ? VI_LABELS[text] : fallback;
    }
  });

  refreshStreamTileLanguage();
}

function setDynLabel(el, label) {
  if (!el) return;
  el.setAttribute('data-dyn-label', label);
  el.textContent = uiLabel(label);
}

function refreshStreamTileLanguage() {
  streams.forEach(inst => {
    const id = inst.id;
    setDynLabel(document.getElementById(`tile-audio-${id}`), inst.audioMuted ? 'Audio Off' : 'Audio On');
    const stopBtn = document.getElementById(`tile-stopbtn-${id}`);
    if (stopBtn) stopBtn.textContent = inst.active ? `■ ${uiLabel('Stop')}` : `▶ ${currentLanguage === 'vi' ? 'Bắt Đầu' : 'Start'}`;
    const sidebar = document.querySelector(`#tile-${id} .tile-sidebar`);
    if (sidebar) {
      sidebar.querySelectorAll('.tile-p-lbl[data-dyn-label], [data-dyn-label]').forEach(el => {
        const label = el.getAttribute('data-dyn-label');
        if (label) el.textContent = uiLabel(label);
      });
      const perfBtn = document.getElementById(`tile-perf-toggle-${id}`);
      const perf = document.getElementById(`tile-perf-${id}`);
      if (perfBtn && perf) perfBtn.textContent = perf.classList.contains('collapsed') ? uiLabel('Show') : uiLabel('Hide');
      ['param', 'det'].forEach(section => {
        const btn = document.getElementById(`tile-${section}-toggle-${id}`);
        const el = document.getElementById(`tile-${section}-section-${id}`);
        if (btn && el) btn.textContent = el.classList.contains('section-collapsed') ? uiLabel('Show') : uiLabel('Hide');
      });
      const closeBtn = sidebar.querySelector('.tile-sb-actions .tile-edit-btn');
      if (closeBtn) closeBtn.textContent = uiLabel('Close');
      const reconnectBtn = sidebar.querySelector('.tile-apply-btn[onclick^="reconnectStreamSource"]');
      if (reconnectBtn) reconnectBtn.textContent = uiLabel('Reconnect Source');
      const applyBtn = sidebar.querySelector('.tile-apply-btn[onclick^="applyStreamParams"]');
      if (applyBtn) applyBtn.textContent = `⟳ ${uiLabel('Apply')}`;
    }
    renderTileDetectionList(inst);
  });
}

// Map of all active streams keyed by string id
const streams = new Map();

/* ── Multi-model helpers ──────────────────────────────────────── */
function getSelectedModels() {
  return [...document.querySelectorAll('input[name="d-model-cb"]:checked')].map(cb => cb.value);
}

function getSelectedAddStreamModels() {
  return [...document.querySelectorAll('input[name="ns-model-cb"]:checked')].map(cb => cb.value);
}

function getSelectedTileModels(id) {
  return [...document.querySelectorAll(`input[name="tm-${id}-cb"]:checked`)].map(cb => cb.value);
}

function selectAllModels(checked) {
  document.querySelectorAll('input[name="d-model-cb"]').forEach(cb => cb.checked = checked);
  _updateModelCount();
}

function selectAllAddStreamModels(checked) {
  document.querySelectorAll('input[name="ns-model-cb"]').forEach(cb => cb.checked = checked);
  _updateAddStreamModelCount();
}

function _updateAddStreamModelCount() {
  const sel = getSelectedAddStreamModels();
  const cnt = document.getElementById('ns-model-count');
  const lbl = document.getElementById('ns-model-select-label');
  if (cnt) {
    cnt.textContent = sel.length ? `${sel.length} selected` : '0 selected';
  }
  if (lbl) {
    if (sel.length === 0) lbl.textContent = 'Select models...';
    else if (sel.length <= 2) lbl.textContent = sel.join(', ');
    else lbl.textContent = `${sel.length} models selected`;
  }
}

function _updateModelCount() {
  const sel = getSelectedModels();
  const cnt = document.getElementById('d-model-count');
  const lbl = document.getElementById('d-model-select-label');
  if (cnt) {
    cnt.textContent = sel.length ? `${sel.length} selected` : '0 selected';
  }
  if (lbl) {
    if (sel.length === 0) lbl.textContent = 'Select models...';
    else if (sel.length <= 2) lbl.textContent = sel.join(', ');
    else lbl.textContent = `${sel.length} models selected`;
  }

  const runBtn = document.getElementById('run-btn');
  if (runBtn && detectMode !== 'webcam') runBtn.disabled = sel.length === 0 || !currentFile;
}

function toggleModelDropdown(prefix, event) {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  const menu = document.getElementById(`${prefix}-model-select-menu`);
  const arrow = document.getElementById(`${prefix}-model-select-arrow`);
  if (!menu) return;
  const isHidden = menu.style.display === 'none' || menu.classList.contains('hidden');
  document.querySelectorAll('[id$="-model-select-menu"]').forEach(m => {
    m.classList.add('hidden');
    m.style.display = 'none';
  });
  document.querySelectorAll('[id$="-model-select-arrow"]').forEach(a => a.style.transform = '');
  if (isHidden) {
    menu.classList.remove('hidden');
    menu.style.display = 'flex';
    if (arrow) arrow.style.transform = 'rotate(180deg)';
  }
}

document.addEventListener('click', (e) => {
  if (!e.target.closest('[id$="-model-select-wrap"]') && !e.target.closest('#modal-restrict-models-list')) {
    document.querySelectorAll('[id$="-model-select-menu"]').forEach(m => {
      m.classList.add('hidden');
      m.style.setProperty('display', 'none', 'important');
    });
    document.querySelectorAll('[id$="-model-select-arrow"]').forEach(a => a.style.transform = '');
  }
});

function selectAllKeyModels(state) {
  const container = document.getElementById('keys-model-list-inner');
  if (container) {
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = state);
    _updateKeyModelCount();
  }
}

function _updateKeyModelCount() {
  const container = document.getElementById('keys-model-list-inner');
  if (!container) return;
  const checked = Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
  const countEl = document.getElementById('keys-model-count');
  const labelEl = document.getElementById('keys-model-select-label');
  if (countEl) countEl.textContent = `${checked.length} selected`;
  if (labelEl) {
    if (checked.length === 0) labelEl.textContent = 'Select models...';
    else if (checked.length <= 2) labelEl.textContent = checked.join(', ');
    else labelEl.textContent = `${checked.length} models selected`;
  }
}

function setupDragSelection(containerId, itemSelector, chkSelector, updateCallback) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  let isDragging = false;
  let startX = 0, startY = 0;
  let marquee = null;
  let initialStates = new Map();

  container.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    if (e.target.closest('button, input, select, a')) return;
    
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    
    const items = container.querySelectorAll(itemSelector);
    items.forEach(item => {
      const chk = item.querySelector(chkSelector);
      if (chk) initialStates.set(item, chk.checked);
    });

    marquee = document.createElement('div');
    marquee.style.cssText = 'position:fixed;border:1px solid #3b82f6;background:rgba(59,130,246,0.18);pointer-events:none;z-index:9999;border-radius:4px;box-shadow:0 0 8px rgba(0,0,0,0.15);';
    marquee.style.left = startX + 'px';
    marquee.style.top = startY + 'px';
    marquee.style.width = '0px';
    marquee.style.height = '0px';
    document.body.appendChild(marquee);
    
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging || !marquee) return;
    
    const currentX = e.clientX;
    const currentY = e.clientY;
    
    const left = Math.min(startX, currentX);
    const top = Math.min(startY, currentY);
    const width = Math.abs(currentX - startX);
    const height = Math.abs(currentY - startY);
    
    marquee.style.left = left + 'px';
    marquee.style.top = top + 'px';
    marquee.style.width = width + 'px';
    marquee.style.height = height + 'px';
    
    const mRect = { left, top, right: left + width, bottom: top + height };
    
    const items = container.querySelectorAll(itemSelector);
    items.forEach(item => {
      const iRect = item.getBoundingClientRect();
      const isIntersecting = !(
        iRect.right < mRect.left ||
        iRect.left > mRect.right ||
        iRect.bottom < mRect.top ||
        iRect.top > mRect.bottom
      );
      const chk = item.querySelector(chkSelector);
      if (chk) {
        chk.checked = isIntersecting ? true : (initialStates.get(item) || false);
        if (item.classList.contains('trk-card') && typeof trkUpdateCardStyle === 'function') {
          trkUpdateCardStyle(item, chk.checked);
        }
      }
    });
    if (updateCallback) updateCallback();
  });

  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      if (marquee) { marquee.remove(); marquee = null; }
      initialStates.clear();
      if (updateCallback) updateCallback();
    }
  });
}

const COLORS = ['#4ade80','#60a5fa','#f87171','#fbbf24','#a78bfa','#34d399','#f472b6','#fb923c','#a3e635','#38bdf8'];
const SYNC_FRAME_HOLD_MS = 5000;

/* ══════════════════════════════════════════════════════════════
   UTILITY
══════════════════════════════════════════════════════════════ */
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function hexToRgb(hex) {
  const n = parseInt(hex.replace('#',''), 16);
  return [(n>>16)&255, (n>>8)&255, n&255];
}

function base64ToArrayBuffer(b64) {
  const binary = atob(b64 || '');
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function maskCacheKey(a, canvasW, canvasH, color) {
  const seg = a.segmentation || {};
  const counts = typeof seg.counts === 'string'
    ? seg.counts
    : Array.isArray(seg.counts) ? seg.counts.join(',') : '';
  return [
    canvasW, canvasH, color, a.source_model || '', a.category_id ?? '',
    a.score != null ? Number(a.score).toFixed(3) : '',
    (a.bbox || []).map(v => Math.round(Number(v) || 0)).join(','),
    Array.isArray(seg.size) ? seg.size.join('x') : '',
    counts.length, counts.slice(0, 24), counts.slice(-24),
  ].join('|');
}

/* ══════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════ */
window.onload = () => {
  const hostInput = document.getElementById('host-input');
  if (hostInput) {
    hostInput.value = window.location.origin;
    HOST = window.location.origin;
  }
  normalizeButtons();
  document.addEventListener('click', normalizeClickedButton, true);
  if (hostInput) {
    hostInput.addEventListener('change', e => {
      HOST = e.target.value.replace(/\/$/, '');
    });
  }
  document.getElementById('modal-overlay').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeModal();
  });
  document.getElementById('add-stream-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeAddStream();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (document.getElementById('add-stream-modal')?.classList.contains('open')) {
        closeAddStream();
      }
      if (document.getElementById('modal-overlay')?.classList.contains('open')) {
        closeModal();
      }
    }
  });
  const savedLang = localStorage.getItem('admin_lang') || 'en';
  const langSel = document.getElementById('lang-select');
  if (langSel) langSel.value = savedLang;
  setLanguage(savedLang, false);
  onSourceChange({ enumerate: false });
  checkHealth();
};

function normalizeButtons(root = document) {
  root.querySelectorAll?.('button:not([type])').forEach(btn => {
    btn.type = 'button';
  });
}

function normalizeClickedButton(e) {
  const btn = e.target?.closest?.('button');
  if (btn && !btn.hasAttribute('type')) btn.type = 'button';
}

window.addEventListener('beforeunload', () => {
  const host = (document.getElementById('host-input')?.value || HOST).replace(/\/$/, '');
  streams.forEach(inst => {
    if (inst.type === 'server_rtsp' && inst.managedStreamId) {
      try {
        fetch(host + `/streams/${inst.managedStreamId}`, {
          method: 'DELETE',
          keepalive: true,
        });
      } catch {}
    }
  });
});

/* ══════════════════════════════════════════════════════════════
   THEME
══════════════════════════════════════════════════════════════ */
function applyTheme(t) {
  const html = document.documentElement;
  html.setAttribute('data-theme', t);
  if (t === 'dark') {
    html.classList.add('dark');
    html.classList.remove('light');
    document.querySelectorAll('#icon-moon, #login-icon-moon').forEach(el => el?.classList.add('hidden'));
    document.querySelectorAll('#icon-sun, #login-icon-sun').forEach(el => el?.classList.remove('hidden'));
  } else {
    html.classList.remove('dark');
    html.classList.add('light');
    document.querySelectorAll('#icon-moon, #login-icon-moon').forEach(el => el?.classList.remove('hidden'));
    document.querySelectorAll('#icon-sun, #login-icon-sun').forEach(el => el?.classList.add('hidden'));
  }
}

function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.classList.contains('dark') || html.getAttribute('data-theme') === 'dark';
  const t = isDark ? 'light' : 'dark';
  applyTheme(t);
  try {
    localStorage.setItem('triton_theme', t);
  } catch (e) {}
}

document.addEventListener('DOMContentLoaded', () => {
  const currentTheme = localStorage.getItem('triton_theme') || (document.documentElement.getAttribute('data-theme') || 'light');
  applyTheme(currentTheme);
});

/* ══════════════════════════════════════════════════════════════
   NAVIGATION
══════════════════════════════════════════════════════════════ */
function switchPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
  const _pg = document.getElementById('page-' + name); if (_pg) _pg.classList.add('active');
  
  let navTarget = name;
  if (['models', 'ensemble', 'system', 'accounts'].includes(name)) {
    navTarget = 'system';
  }
  if (['stream', 'detect'].includes(name)) {
    navTarget = 'stream';
  }
  
  document.querySelectorAll('.sidebar-btn').forEach(b => {
    if ((b.getAttribute('onclick') || '').includes(`'${navTarget}'`)) b.classList.add('active');
  });
  if (['models','ensemble'].includes(name)) loadModels();
  if (name === 'models') loadGPUs();
  if (name === 'system') { checkHealth(); loadGPUs(); loadSystemStatus(); }
  if (name === 'keys') { loadApiKeys(); }
  if (name === 'accounts') { loadAccounts(); }
}

function openAddStreamPreset(source) {
  switchPage('stream');
  if (typeof openAddStream === 'function') openAddStream('stream');
  const sourceSelect = document.getElementById('ns-source');
  if (sourceSelect) {
    sourceSelect.value = source;
    if (typeof onSourceChange === 'function') onSourceChange();
  }
}

/* ══════════════════════════════════════════════════════════════
   TOAST / MODAL
══════════════════════════════════════════════════════════════ */
function toast(msg, type = 'info', duration = 3500) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.getElementById('toast-container').appendChild(t);
  setTimeout(() => t.remove(), duration);
}

function openModal(title, body, onConfirm) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').innerHTML = body;
  document.getElementById('modal-confirm-btn').style.display = '';
  document.getElementById('modal-confirm-btn').onclick = () => { onConfirm(); closeModal(); };
  document.getElementById('modal-overlay').classList.add('open');
}
function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
  document.getElementById('modal-confirm-btn').style.display = '';
}

/* ══════════════════════════════════════════════════════════════
   API
══════════════════════════════════════════════════════════════ */
async function apiFetch(path, opts = {}) {
  HOST = document.getElementById('host-input').value.replace(/\/$/, '');
  // Prevent Chrome from serving stale cached API responses across sessions
  if (!opts.method || opts.method.toUpperCase() === 'GET') {
    opts.cache = 'no-store';
  }
  try {
    const r = await fetch(HOST + path, opts);
    if (r.status === 401) {
      const error = new Error('Unauthorized');
      error.status = 401;
      throw error;
    }
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      const detail = err.detail || r.statusText;
      const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      error.status = r.status;
      throw error;
    }
    return r.json();
  } catch (e) {
    const msg = e.message || '';
    const isIgnore = e.status === 401 || 
                     msg === 'Unauthorized' || 
                     msg === 'Failed to fetch' || 
                     msg.includes('NetworkError') || 
                     e.name === 'AbortError';
    if (!isIgnore) {
      toast('API Error: ' + msg, 'error');
    }
    throw e;
  }
}

function _modelMeta(name) {
  return modelInfoCache[name] || [...allModels, ...allEnsembles].find(m => m.name === name) || {};
}

function modelRequiresPrompts(name) {
  const m = _modelMeta(name);
  return !!(
    m.type === 'yoloe-dynamic' ||
    m.requires_prompts ||
    m.ensemble_requires_prompts ||
    m.ensemble_kind === 'hybrid' ||
    name.toLowerCase().includes('yoloe')
  );
}

function missingPromptModels(models, promptText) {
  return models.filter(m => modelRequiresPrompts(m) && !promptText);
}

function splitLiveTextParams(models, text) {
  const value = (text || '').trim();
  if (!value) return { classes: null, prompts: null };
  return models.some(model => modelRequiresPrompts(model))
    ? { classes: null, prompts: value }
    : { classes: value, prompts: null };
}

function appendPromptIfNeeded(url, model, promptText) {
  return promptText && modelRequiresPrompts(model)
    ? url + `&prompts=${encodeURIComponent(promptText)}`
    : url;
}

function appendPromptFormIfNeeded(form, model, promptText) {
  if (promptText && modelRequiresPrompts(model)) form.append('prompts', promptText);
}

function annColorKey(a) {
  return `${a.source_model || ''}:${a.category_name || (a.category_id ?? 'unknown')}`;
}

function annLabel(a) {
  return a.category_name || `class_${a.category_id ?? '?'}`;
}

function infoMark(text) {
  return '';
}

function hintLabel(label, text) {
  const shown = currentLanguage === 'vi' ? (VI_LABELS[label] || label) : label;
  const title = uiTitle(text);
  return `<span class="hint-text" data-dyn-label="${escAttr(label)}" data-title-default="${escAttr(text)}" title="${escHtml(title)}">${escHtml(shown)}</span>`;
}

function normalizeAnnotations(anns, fallbackModel) {
  return (anns || []).map(a => {
    const out = {...a};
    if (!out.source_model && fallbackModel) out.source_model = fallbackModel;
    if (!out.category_name) out.category_name = `class_${out.category_id ?? '?'}`;
    delete out._maskCanvas;
    return out;
  });
}

function _fmtMs(v) {
  if (v == null || v === '') return '—';
  if (typeof v === 'string' && v.trim().endsWith('ms')) return v.trim();
  const n = Number(v);
  return Number.isFinite(n) ? `${n.toFixed(n >= 10 ? 0 : 1)}ms` : `${v}ms`;
}

function _fmtBandwidth(bytesPerSec) {
  const n = Number(bytesPerSec);
  if (!Number.isFinite(n) || n <= 0) return '—';
  const mbps = n * 8 / 1_000_000;
  if (mbps >= 1) return `${mbps.toFixed(mbps >= 10 ? 1 : 2)} Mbps`;
  const kbps = n * 8 / 1000;
  return `${kbps.toFixed(kbps >= 10 ? 0 : 1)} Kbps`;
}

async function drawJpegBytesToCanvas(canvas, data) {
  const blob = data instanceof Blob ? data : new Blob([data], {type:'image/jpeg'});
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(blob);
      if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
      if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      const size = { width: bitmap.width, height: bitmap.height };
      bitmap.close();
      return size;
    } catch {}
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      URL.revokeObjectURL(url);
      if (canvas.width !== img.naturalWidth) canvas.width = img.naturalWidth;
      if (canvas.height !== img.naturalHeight) canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Could not decode JPEG preview frame'));
    };
    img.src = url;
  });
}

async function drawSyncedJpegWithAnnotations(canvas, data, anns, imgShape, catMap) {
  if (!canvas || !data || !imgShape) return null;
  try {
    await drawJpegBytesToCanvas(canvas, data);
    const ctx = canvas.getContext('2d');
    if (ctx && anns?.length) drawAnnotations(ctx, canvas.width, canvas.height, anns, imgShape, catMap);
    const frozen = document.createElement('canvas');
    frozen.width = canvas.width;
    frozen.height = canvas.height;
    frozen.getContext('2d').drawImage(canvas, 0, 0);
    return frozen;
  } catch {
    return null;
  }
}

function getDetectOverlayMode() {
  const value = document.getElementById('d-overlay-mode')?.value || 'exact';
  return value === 'native_exact' ? 'native_exact' : 'exact';
}

function detectExactModeEnabled() {
  return getDetectOverlayMode() === 'exact';
}

function overlayModeLabel(mode) {
  return mode === 'native_exact' ? 'Native FPS Live' : 'Exact Boxes';
}

function normalizeOverlayMode(value) {
  return value === 'native_exact' ? 'native_exact' : 'exact';
}

function nativeExactModeEnabled(mode) {
  return normalizeOverlayMode(mode) === 'native_exact';
}

function alignedBoxesModeEnabled(mode) {
  return normalizeOverlayMode(mode) === 'exact';
}

function getDetectLiveFps(fallback = 10) {
  return getFpsInputValue('d-fps', fallback);
}

function onDetectOverlayModeChange() {
  clampFpsInput(document.getElementById('d-fps'), 10, false);
  detectWebcamExactFrame = null;
  detectWebcamExactFrameAt = 0;
  videoDetectExactFrame = null;
  videoDetectExactFrameAt = 0;
}

function onStreamOverlayModeChange() {
  clampFpsInput(document.getElementById('ns-fps'), 10, false);
  clampFpsInput(document.getElementById('ns-preview-fps'), 10, false);
}

function _rtspBackendText(data = {}) {
  const backend = data.rtsp_backend || '—';
  const requested = data.rtsp_backend_requested;
  const gst = data.rtsp_gstreamer_available;
  const decoder = data.rtsp_gstreamer_decoder;
  const gpu = data.nvidia_gpu_detected;
  const bits = [`backend ${backend}`];
  if (requested && requested !== backend) bits.push(`requested ${requested}`);
  if (gst != null) bits.push(`gst ${gst ? 'yes' : 'no'}`);
  if (decoder) bits.push(`decoder ${decoder}`);
  if (gpu != null) bits.push(`gpu ${gpu ? 'yes' : 'no'}`);
  return bits.join(' · ');
}

function normalizeRtspBackendChoice(value, notify = false) {
  const backend = (value || 'auto').toLowerCase();
  if (['auto', 'gstreamer', 'opencv'].includes(backend)) return backend;
  if (notify) toast('Unsupported RTSP backend. Using Auto/GStreamer instead.', 'error', 6000);
  return 'auto';
}

function rtspBackendOptionsHtml(selected = 'auto') {
  const sel = normalizeRtspBackendChoice(selected);
  return `
    <option value="auto" ${sel === 'auto' ? 'selected' : ''}>Auto</option>
    <option value="gstreamer" ${sel === 'gstreamer' ? 'selected' : ''}>GStreamer</option>
    <option value="opencv" ${sel === 'opencv' ? 'selected' : ''}>OpenCV / FFmpeg</option>`;
}

function sourceSizeOptionsHtml(selected = 720) {
  const value = String(selected == null ? 720 : selected);
  return [
    ['0', 'Native'],
    ['512', '512p'],
    ['640', '640p'],
    ['720', '720p'],
    ['960', '960p'],
    ['1080', '1080p'],
    ['1440', '1440p'],
  ].map(([v, label]) => `<option value="${v}" ${String(v) === value ? 'selected' : ''}>${label}</option>`).join('');
}

function updateRtspBackendSelects() {
  document.querySelectorAll('select.rtsp-backend-select, #ns-rtsp-backend-select').forEach(sel => {
    const value = normalizeRtspBackendChoice(sel.value);
    sel.innerHTML = rtspBackendOptionsHtml(value);
    sel.value = value;
    sel.title = 'Use Auto unless testing a specific RTSP backend.';
  });
}

function timingStatsHtml(data, opts = {}) {
  const timing = data?.timing_ms || {};
  const serverTotal = timing.total ?? timing.inference ?? timing.elapsed;
  const total = opts.totalMs ?? serverTotal;
  const imgsz = data?.inference_imgsz || opts.imgsz || [];
  const shape = data?.image_shape || opts.imageShape || [];
  if (opts.live) {
    return `
      <div class="infer-stat live">mode: <span>live</span></div>
      <div class="infer-stat">latency: <span>${_fmtMs(opts.clientMs ?? total)}</span></div>
      <div class="infer-stat">server: <span>${_fmtMs(serverTotal)}</span></div>
      ${timing.triton != null ? `<div class="infer-stat">triton: <span>${_fmtMs(timing.triton)}</span></div>` : ''}
      ${timing.postprocess != null ? `<div class="infer-stat">post: <span>${_fmtMs(timing.postprocess)}</span></div>` : ''}
      ${opts.resultFps != null ? `<div class="infer-stat">result fps: <span>${opts.resultFps}</span></div>` : ''}
      ${opts.requestedFps != null ? `<div class="infer-stat">cap: <span>${opts.requestedFps}</span></div>` : ''}
      <div class="infer-stat">imgsz: <span>${Array.isArray(imgsz) ? imgsz.join('×') : imgsz}</span></div>
      <div class="infer-stat">shape: <span>${Array.isArray(shape) ? shape.join('×') : shape}</span></div>`;
  }
  return `
    <div class="infer-stat">time: <span>${_fmtMs(total)}</span></div>
    ${timing.triton != null ? `<div class="infer-stat">triton: <span>${_fmtMs(timing.triton)}</span></div>` : ''}
    ${timing.postprocess != null ? `<div class="infer-stat">post: <span>${_fmtMs(timing.postprocess)}</span></div>` : ''}
    ${opts.fps != null ? `<div class="infer-stat">fps: <span>${opts.fps}</span></div>` : ''}
    <div class="infer-stat">imgsz: <span>${Array.isArray(imgsz) ? imgsz.join('×') : imgsz}</span></div>
    <div class="infer-stat">shape: <span>${Array.isArray(shape) ? shape.join('×') : shape}</span></div>`;
}

function perModelTimingStatsHtml(perModel) {
  if (!perModel || !Object.keys(perModel).length) return '';
  return Object.entries(perModel).map(([name, timing]) => {
    const total = timing?.total ?? timing?.inference ?? timing?.triton;
    return `<div class="infer-stat">model ${escHtml(name)}: <span>${_fmtMs(total)}</span></div>`;
  }).join('');
}

function resetDetectLivePerf(mode = '') {
  detectLivePerf = { mode, windowStart: performance.now(), frames: 0, measuredFps: 0 };
}

function noteDetectLiveResult(mode) {
  const now = performance.now();
  if (detectLivePerf.mode !== mode || !detectLivePerf.windowStart) resetDetectLivePerf(mode);
  detectLivePerf.frames += 1;
  const elapsed = now - detectLivePerf.windowStart;
  if (elapsed >= 1000) {
    detectLivePerf.measuredFps = Math.round((detectLivePerf.frames * 1000) / elapsed);
    detectLivePerf.windowStart = now;
    detectLivePerf.frames = 0;
  }
  return detectLivePerf.measuredFps || Math.max(1, Math.round((detectLivePerf.frames * 1000) / Math.max(1, elapsed)));
}

function updateDetectLiveStats(data, stats = {}) {
  const el = document.getElementById('detect-stats');
  const opts = typeof stats === 'number'
    ? { requestedFps: stats }
    : stats;
  if (el) el.innerHTML = timingStatsHtml(data, { live:true, ...opts });
}

function clampFpsValue(value, fallback = 10, notify = false) {
  const raw = parseInt(value);
  const fps = Number.isFinite(raw) && raw > 0 ? raw : fallback;
  const max = Number.isFinite(apiMaxFps) ? apiMaxFps : 30;
  if (max <= 0) return Math.max(1, fps);
  const clamped = Math.max(1, Math.min(fps, max));
  if (notify && fps > max) toast(`FPS capped to API max_fps (${max})`, 'info');
  return clamped;
}

function clampFpsInput(el, fallback = 10, notify = false) {
  if (!el) return fallback;
  const fps = clampFpsValue(el.value, fallback, notify);
  if (Number.isFinite(apiMaxFps) && apiMaxFps > 0) el.max = String(apiMaxFps);
  else el.removeAttribute('max');
  el.value = String(fps);
  return fps;
}

function applyApiFpsLimit(maxFps) {
  const n = parseInt(maxFps);
  if (Number.isFinite(n)) apiMaxFps = n;
  ['d-fps', 'ns-fps', 'ns-preview-fps'].forEach(id => clampFpsInput(document.getElementById(id), id === 'd-fps' ? 10 : apiMaxFps));
  document.querySelectorAll('input[id^="tf-"]').forEach(el => clampFpsInput(el, apiMaxFps));
  document.querySelectorAll('input[id^="tpf-"]').forEach(el => clampFpsInput(el, apiMaxFps));
  streams.forEach(inst => {
    if (apiMaxFps > 0 && inst.fps > apiMaxFps) {
      inst.fps = apiMaxFps;
      const el = document.getElementById(`tf-${inst.id}`);
      if (el) el.value = String(apiMaxFps);
      if (inst.inferWsList?.length) {
        inst.inferWsList.forEach(w => w.close());
        inst.inferWsList = [];
        inst.inferWs = null;
        inst._connectInfer();
      }
    }
  });
  const detectLbl = document.getElementById('d-fps-limit');
  if (detectLbl) detectLbl.textContent = apiMaxFps > 0 ? `API max: ${apiMaxFps}` : 'API max: unlimited';
  const streamLbl = document.getElementById('ns-fps-limit');
  if (streamLbl) streamLbl.textContent = apiMaxFps > 0 ? `API max: ${apiMaxFps}` : 'API max: unlimited';
}

function getFpsInputValue(id, fallback = 10) {
  return clampFpsInput(document.getElementById(id), fallback, true);
}

/* ══════════════════════════════════════════════════════════════
   HEALTH
══════════════════════════════════════════════════════════════ */
async function checkHealth() {
  HOST = document.getElementById('host-input').value.replace(/\/$/, '');
  try {
    const h = await apiFetch('/health');
    apiHealthCache = h;
    const dot = document.getElementById('triton-dot');
    const lbl = document.getElementById('triton-label');
    dot.className = h.triton_ready ? 'status-dot ok' : 'status-dot err';
    lbl.textContent = h.triton_ready 
      ? (currentLanguage === 'vi' ? 'Triton Sẵn Sàng' : 'Triton Online')
      : (currentLanguage === 'vi' ? 'Triton Lỗi' : 'Triton Offline');
    applyApiFpsLimit(h.max_fps);
    renderHealthCards(h);
    updateRtspBackendUi(h);
    updateRtspBackendSelects();
    // Disabled noisy Triton ready toast
    loadModels();
    loadGPUs();
  } catch {
    document.getElementById('triton-dot').className = 'status-dot err';
    document.getElementById('triton-label').textContent = currentLanguage === 'vi' ? 'Ngoại Tuyến' : 'Offline';
  }
}

function updateRtspBackendUi(h) {
  apiHealthCache = h || apiHealthCache;
  const el = document.getElementById('ns-rtsp-backend');
  if (el) el.textContent = `Backend: ${_rtspBackendText(h)}`;
  const streamEl = document.getElementById('stream-rtsp-backend');
  if (streamEl) streamEl.textContent = _rtspBackendText(h);
  updateRtspBackendSelects();
}

function renderHealthCards(h) {
  const g = document.getElementById('health-grid');
  const cards = [
    { label:'Status', val:h.status||'—', cls:h.status==='ok'?'ok':'err', hint:'Overall API health.' },
    { label:'Triton Ready', val:h.triton_ready?'YES':'NO', cls:h.triton_ready?'ok':'err', hint:'Whether NVIDIA Triton server is reachable and ready to serve models.' },
    { label:'Encoder', val:h.encoder_loaded?'LOADED':'NONE', cls:h.encoder_loaded?'ok':'warn', hint:'YOLOE text encoder status. Needed for prompt-based YOLOE models.' },
    { label:'Max Concurrent', val:h.max_concurrent??'—', cls:'', hint:'Maximum concurrent inference requests allowed by the API server.' },
    { label:'Max FPS', val:h.max_fps??'—', cls:'', hint:'API-wide FPS cap used to clamp browser live inference and RTSP stream settings.' },
    { label:'GPUs', val:h.gpu_count??'—', cls:'', hint:'Number of NVIDIA GPUs visible to the API container.' },
    { label:'RTSP Backend', val:(h.rtsp_backend || '—').toUpperCase(), cls:h.rtsp_backend==='gstreamer'?'ok':'warn', hint:'Actual backend used by API to read RTSP streams.' },
    { label:'RTSP Requested', val:(h.rtsp_backend_requested || '—').toUpperCase(), cls:'', hint:'Configured RTSP backend preference from environment.' },
    { label:'GStreamer', val:h.rtsp_gstreamer_available?'YES':'NO', cls:h.rtsp_gstreamer_available?'ok':'warn', hint:'Whether OpenCV in this container can use GStreamer pipelines.' },
    { label:'GST Decoder', val:(h.rtsp_gstreamer_decoder || '—').toUpperCase(), cls:h.rtsp_gstreamer_decoder?.startsWith('nv')?'ok':'', hint:'Decoder selected by GStreamer auto mode. NVIDIA decoders reduce CPU video decode work.' },
    { label:'go2rtc', val:h.go2rtc_enabled ? (h.go2rtc_ready ? 'READY' : 'DOWN') : 'OFF', cls:h.go2rtc_ready?'ok':(h.go2rtc_enabled?'err':'warn'), hint:'Live-view fan-out service. RTSP Native FPS Live uses go2rtc WebRTC; API inference can read the go2rtc RTSP restream.' },
  ];
  g.innerHTML = cards.map(c => `
    <div class="health-card" title="${escHtml(c.hint)}">
      <div class="health-card-label">${c.label}</div>
      <div class="health-card-value ${c.cls}">${c.val}</div>
    </div>`).join('');
}

/* ══════════════════════════════════════════════════════════════
   GPUs
══════════════════════════════════════════════════════════════ */
async function loadGPUs() {
  try {
    const g = await apiFetch('/gpus');
    gpuInfoCache = g || { gpus: [], models_by_gpu: {}, default_gpu: null };
    renderUploadGpuList();
    const tbody = document.getElementById('gpu-tbody');
    if (!g.gpus?.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text3);padding:20px;">No GPU data</td></tr>';
      return;
    }
    tbody.innerHTML = g.gpus.map(gpu => `<tr>
      <td>${gpu.index}${g.default_gpu===gpu.index?' <span class="badge badge-green">default</span>':''}</td>
      <td>${gpu.name}</td>
      <td>${gpu.memory_total_mb ? (gpu.memory_total_mb/1024).toFixed(1)+' GB' : '—'}</td>
      <td>${(g.models_by_gpu?.[gpu.index]||[]).join(', ')||'—'}</td>
    </tr>`).join('');
    renderSimpleGpuList();
  } catch {}
}

function renderUploadGpuList(selected = null) {
  const wrap = document.getElementById('up-gpu-list');
  if (!wrap) return;
  const gpus = gpuInfoCache.gpus || [];
  if (!gpus.length) {
    wrap.innerHTML = '<div class="model-check-empty">No GPU data yet</div>';
    return;
  }
  const selectedSet = selected || new Set([String(gpuInfoCache.default_gpu ?? gpus[0]?.index ?? 0)]);
  wrap.innerHTML = gpus.map(gpu => {
    const idx = String(gpu.index);
    const vram = gpu.memory_total_mb ? `${(gpu.memory_total_mb / 1024).toFixed(1)} GB` : 'VRAM unknown';
    return `
      <label class="upload-gpu-item" title="${escHtml(gpu.name || 'Unknown GPU')} · ${vram}">
        <input type="checkbox" name="up-gpu" value="${idx}" ${selectedSet.has(idx) ? 'checked' : ''} />
        <span>GPU ${idx}${gpu.index === gpuInfoCache.default_gpu ? ' · default' : ''} · ${escHtml(gpu.name || 'Unknown')} · ${vram}</span>
      </label>`;
  }).join('');
}

function getSelectedUploadGpus() {
  return [...document.querySelectorAll('input[name="up-gpu"]:checked')].map(cb => cb.value);
}

function parseUploadLabels() {
  const raw = document.getElementById('up-labels')?.value || '';
  return raw.split(/[\n,]/).map(s => s.trim()).filter(Boolean);
}

/* ══════════════════════════════════════════════════════════════
   SYSTEM STATUS / MONITOR
══════════════════════════════════════════════════════════════ */
let _sysmonTimer = null;
let _sysmonIntervalMs = 10000;
async function loadSystemStatus() {
  clearTimeout(_sysmonTimer);
  try {
    const s = await apiFetch('/system/status');
    _renderSysmon(s);
    _renderTritonStats(s.triton_metrics_summary);
  } catch {
    const g = document.getElementById('sysmon-grid');
    if (g) g.innerHTML = '<div class="empty-state">Could not load — check connection</div>';
  }
  if (document.getElementById('page-system')?.classList.contains('active') && _sysmonIntervalMs > 0) {
    _sysmonTimer = setTimeout(loadSystemStatus, _sysmonIntervalMs);
  }
}

function setSysmonInterval(ms) {
  _sysmonIntervalMs = ms;
  document.querySelectorAll('.sysmon-rate-btn').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.trim() === `${ms / 1000}s`);
  });
  loadSystemStatus();
}

function _barCls(pct) { return pct >= 85 ? 'crit' : pct >= 60 ? 'warn' : 'ok'; }

function _renderSysmon(s) {
  const g = document.getElementById('sysmon-grid');
  if (!g) return;
  const cards = [];
  if (s.host?.cpu) {
    const p = s.host.cpu.percent ?? 0;
    cards.push({ label:`CPU (${s.host.cpu.count||'?'} cores)`, val:`${p.toFixed(1)}%`, sub:'utilization', pct:p });
  }
  if (s.host?.memory) {
    const m = s.host.memory;
    cards.push({ label:'RAM', val:`${m.percent.toFixed(1)}%`,
      sub:`${(m.used_mb/1024).toFixed(1)} / ${(m.total_mb/1024).toFixed(1)} GB`, pct:m.percent });
  }
  (s.gpus||[]).forEach(gpu => {
    const util = gpu.gpu_util_percent ?? 0;
    const vUsed = gpu.memory_used_mb ?? 0, vTot = gpu.memory_total_mb ?? 0;
    const vPct  = vTot ? Math.round(vUsed/vTot*100) : 0;
    const temp  = gpu.temperature_c != null ? `${gpu.temperature_c}°C` : '—';
    cards.push({ label:`GPU ${gpu.index} · ${temp}`, val:`${util}%`,
      sub:`VRAM ${(vUsed/1024).toFixed(1)}/${(vTot/1024).toFixed(1)} GB · ${vPct}%`, pct:util, pct2:vPct });
  });
  if (s.triton?.container_stats && Object.keys(s.triton.container_stats).length > 0) {
    const tc = s.triton.container_stats;
    cards.push({
      label: 'Triton CPU (Container)',
      val: `${tc.cpu_percent.toFixed(1)}%`,
      sub: currentLanguage === 'vi' ? 'Sử dụng CPU của container' : 'Container CPU utilization',
      pct: tc.cpu_percent
    });
    cards.push({
      label: 'Triton RAM (Container)',
      val: `${tc.memory_percent.toFixed(1)}%`,
      sub: `${(tc.memory_used_mb / 1024).toFixed(2)} / ${(tc.memory_limit_mb / 1024).toFixed(1)} GB`,
      pct: tc.memory_percent
    });
  }
  g.innerHTML = cards.length ? cards.map(c => `
    <div class="sysmon-card">
      <div class="sysmon-card-label">${c.label}</div>
      <div class="sysmon-card-val">${c.val}</div>
      <div class="sysmon-card-sub">${c.sub}</div>
      <div class="sysmon-bar-wrap"><div class="sysmon-bar ${_barCls(c.pct)}" style="width:${Math.min(c.pct,100)}%"></div></div>
      ${c.pct2!=null?`<div class="sysmon-bar-wrap"><div class="sysmon-bar ok" style="width:${Math.min(c.pct2,100)}%;"></div></div>`:''}
    </div>`).join('') : '<div class="empty-state">No resource data</div>';
}

function _renderTritonStats(summary) {
  const wrap = document.getElementById('triton-stats-wrap');
  if (!wrap) return;
  const models = summary?.models;
  if (!models || !Object.keys(models).length) {
    wrap.innerHTML = '<div style="color:var(--text3);font-family:var(--font-mono);font-size:11px;padding:6px 0;">No Triton stats</div>';
    return;
  }
  const total = summary.inference_success_total ?? '—';
  const fail  = summary.inference_failure_total ?? '—';
  wrap.innerHTML = `
    <div style="font-family:var(--font-mono);font-size:10px;color:var(--text2);margin-bottom:6px;">
      Total: <span style="color:var(--blue)">${total}</span> success
      · <span style="color:var(--red)">${fail}</span> fail
    </div>
    <table class="triton-stats-tbl">
      <thead><tr><th>Model</th><th>Success</th><th>Fail</th></tr></thead>
      <tbody>${Object.entries(models).map(([name,v])=>`
        <tr>
          <td>${escHtml(name)}</td>
          <td style="color:var(--blue)">${v.success??0}</td>
          <td style="color:${v.failure?'var(--red)':'var(--text3)'}">${v.failure??0}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

/* ══════════════════════════════════════════════════════════════
   MODELS
══════════════════════════════════════════════════════════════ */
async function loadModels() {
  try {
    const data = await apiFetch('/models');
    allModels    = data.single_models   || data.models?.filter(m => m.kind !== 'ensemble') || [];
    allEnsembles = data.ensemble_models || data.models?.filter(m => m.kind === 'ensemble') || [];
    modelInfoCache = {};

    await Promise.all(allEnsembles.map(async ens => {
      try {
        const data = await apiFetch(`/ensemble/${ens.name}`);
        ens.steps = (data.steps || []).map(s =>
          typeof s === 'string' ? { model: s, version: -1 }
          : { model: s.model || s.model_name || '?', version: s.version ?? -1 }
        );
      } catch (e) {
        try {
          const cfg = await apiFetch(`/models/${ens.name}/config`);
          ens.steps = (cfg.ensemble_scheduling?.step || []).map(s => ({
            model: s.model_name || s.model || '?', version: s.model_version ?? -1
          }));
        } catch { ens.steps = []; }
      }
    }));

    const combined = [...allModels, ...allEnsembles];
    await Promise.all(combined.map(async m => {
      try {
        const info = await apiFetch(`/models/${m.name}/info`);
        Object.assign(m, info);
        modelInfoCache[m.name] = {...m, ...info};
      } catch {
        modelInfoCache[m.name] = {...m};
      }
    }));

    await Promise.all(allEnsembles.map(async ens => {
      try {
        const v = await apiFetch(`/ensemble/${ens.name}/validate`);
        ens.valid = v.valid;
        ens.requires_prompts = v.requires_prompts;
        ens.ensemble_requires_prompts = v.requires_prompts;
        ens.not_ready_models = v.not_ready_models || [];
        ens.missing_models = v.missing_models || [];
        modelInfoCache[ens.name] = {...(modelInfoCache[ens.name] || ens), ...ens, ...v, ensemble_requires_prompts: v.requires_prompts};
      } catch {}
    }));

    renderModelsGrid();
    populateModelSelects();
  } catch {}
}

function renderModelsGrid() {
  const grid = document.getElementById('models-grid');
  grid.innerHTML = allModels.length
    ? allModels.map(modelCard).join('')
    : '<div class="empty-state"><div class="empty-icon">⬤</div>No models loaded</div>';
  const eg = document.getElementById('ensemble-grid');
  eg.innerHTML = allEnsembles.length
    ? allEnsembles.map(ensCard).join('')
    : '<div class="empty-state"><div class="empty-icon">⬤</div>No ensembles</div>';
}

function ensCard(m) {
  const steps = m.steps || [];
  const valid = m.valid !== false;
  const nameArg = jsAttr(m.name);
  const promptBadge = m.requires_prompts || m.ensemble_requires_prompts
    ? '<span class="badge badge-yellow">PROMPTS</span>'
    : '';
  const validBadge = valid
    ? '<span class="badge badge-purple model-status-badge">ENSEMBLE</span>'
    : '<span class="badge badge-red model-status-badge">INVALID</span>';
  const stepsHtml = steps.length
    ? steps.map((s, i) => {
        const modelName = typeof s === 'string' ? s : (s.model || s.name || '?');
        return `<div style="display:flex;align-items:center;gap:6px;padding:4px 8px;
          background:var(--bg3);border-radius:var(--radius);border-left:2px solid var(--purple);margin-bottom:3px;">
          <span style="font-family:var(--font-mono);font-size:10px;color:var(--text3);width:16px;flex-shrink:0;">${i+1}.</span>
          <span style="font-family:var(--font-mono);font-size:11px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text);">${escHtml(modelName)}</span>
          <button onclick="event.stopPropagation(); removeStepFromEnsemble('${escHtml(m.name)}',${i})"
            style="border:none;background:transparent;color:var(--red);cursor:pointer;font-size:12px;
            padding:0 2px;opacity:.55;line-height:1;flex-shrink:0;"
            title="Remove step" onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=.55">✕</button>
        </div>`;
      }).join('')
    : `<div style="color:var(--text3);font-family:var(--font-mono);font-size:11px;padding:4px 0 6px;">No steps</div>`;

  const menuId = `ensemble-menu-${safeDomId(m.name)}`;

  return `<div class="model-card cursor-pointer" onclick="editEnsemble('${nameArg}')">
    <div class="model-card-top">
      <div>
        <div class="model-card-title">${escHtml(m.name)}</div>
        <div class="model-card-meta">ensemble · ${steps.length} model${steps.length!==1?'s':''}${m.not_ready_models?.length ? ` · not ready: ${escHtml(m.not_ready_models.join(', '))}` : ''}</div>
      </div>
    </div>
    <div style="margin-bottom:8px;">${stepsHtml}</div>
    <div style="display:flex;align-items:center;gap:5px;margin-bottom:8px;" onclick="event.stopPropagation()">
      <select id="add-step-select-${escHtml(m.name)}" class="param-input"
        style="flex:1;font-size:10px;padding:3px 6px;">
        <option value="">+ Add model to ensemble…</option>
        ${[...allModels,...allEnsembles].filter(x=>x.name!==m.name).map(x=>
          `<option value="${escHtml(x.name)}">${escHtml(x.name)}</option>`
        ).join('')}
      </select>
      <button class="btn btn-ghost" style="padding:3px 10px;font-size:10px;"
        onclick="event.stopPropagation(); addStepToEnsemble('${escHtml(m.name)}')">Add</button>
    </div>
    <div class="model-card-footer">
      <div style="display:flex;align-items:center;gap:4px;">
        ${validBadge}
        ${promptBadge}
      </div>
      <div class="model-card-actions" style="margin-left:auto;" onclick="event.stopPropagation()">
        <div class="model-menu" id="${menuId}">
          <button class="model-icon-btn" onclick="event.stopPropagation(); toggleModelMenu(event, '${menuId}')" title="More actions">${modelIcon('dots')}</button>
          <div class="model-menu-popover">
            <button class="model-menu-item" onclick="event.stopPropagation(); closeModelMenus(); validateEnsemble('${nameArg}')">${modelIcon('check')} Validate</button>
            <button class="model-menu-item danger" onclick="event.stopPropagation(); closeModelMenus(); confirmDeleteEnsemble('${nameArg}')">${modelIcon('trash')} Delete</button>
          </div>
        </div>
      </div>
    </div>
  </div>`;
}

function populateModelSelects() {
  const combined = [...allModels, ...allEnsembles];

  // ── Multi-checkbox list for detect page ──
  const list = document.getElementById('d-model-list');
  const listOverlay = document.getElementById('d-model-list-overlay');
  const targetLists = [list, listOverlay].filter(Boolean);

  targetLists.forEach(l => {
    const prevChecked = new Set(getSelectedModels());
    l.innerHTML = combined.length
      ? combined.map(m => `
        <label class="model-check-item">
          <input type="checkbox" name="d-model-cb" value="${escHtml(m.name)}"
            ${prevChecked.has(m.name) || (prevChecked.size === 0 && combined.length === 1) ? 'checked' : ''} />
          <span style="font-family:var(--font-mono);font-size:11px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(m.name)}</span>
        </label>`).join('')
      : '<span class="model-check-empty">No models — connect first</span>';
    l.querySelectorAll('input[type=checkbox]').forEach(cb =>
      cb.addEventListener('change', _updateModelCount));
  });
  _updateModelCount();

  // ── Multi-checkbox list for Add Stream modal ──
  const streamList = document.getElementById('ns-model-list');
  if (streamList) {
    const prevChecked = new Set(getSelectedAddStreamModels());
    streamList.innerHTML = combined.length
      ? combined.map(m => `
        <label class="model-check-item">
          <input type="checkbox" name="ns-model-cb" value="${escHtml(m.name)}"
            ${prevChecked.has(m.name) || (prevChecked.size === 0 && combined.length === 1) ? 'checked' : ''} />
          <span style="font-family:var(--font-mono);font-size:11px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(m.name)}</span>
        </label>`).join('')
      : '<span class="model-check-empty">No models — connect first</span>';
    streamList.querySelectorAll('input[type=checkbox]').forEach(cb =>
      cb.addEventListener('change', _updateAddStreamModelCount));
    _updateAddStreamModelCount();
  }

  // ── Multi-checkbox list for API Keys modal ──
  const keysList = document.getElementById('keys-model-list-inner');
  if (keysList) {
    const prevChecked = new Set(Array.from(keysList.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value));
    keysList.innerHTML = combined.length
      ? combined.map(m => `
        <label class="flex items-center gap-1.5 text-[11px] text-primary cursor-pointer p-1 rounded hover:bg-secondary">
          <input type="checkbox" class="accent-[#10b981]" value="${escHtml(m.name)}" ${prevChecked.has(m.name) ? 'checked' : ''} onchange="_updateKeyModelCount()" />
          <span>${escHtml(m.name)}</span>
        </label>`).join('')
      : '<span class="text-[10px] text-primary-variant p-1">No active models loaded in Triton.</span>';
    _updateKeyModelCount();
  }

  // ── Other selects (stream modal, config, ensemble) ──
  ['cfg-model','ens-step-select'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const cur = el.value;
    el.innerHTML = id === 'ens-step-select'
      ? '<option value="">Add model…</option>'
      : '<option value="">— select —</option>';
    combined.forEach(m => {
      const o = document.createElement('option');
      o.value = m.name; o.textContent = m.name;
      el.appendChild(o);
    });
    if (cur && Array.from(el.options).some(o => o.value === cur)) {
      el.value = cur;
    }
  });
}

function modelAction(name, action) {
  if (!action) return;
  if (action === 'info') showModelInfo(name);
  else if (action === 'reload') reloadModel(name);
  else if (action === 'refresh') refreshModel(name);
  else if (action === 'delete') confirmDeleteModel(name);
}

function confirmDeleteModel(name) {
  openModal(`Delete model "${name}"?`,
    `<p style="font-family:var(--font-mono);font-size:12px;color:var(--text2);">This will unload and permanently delete the model directory.</p>`,
    () => deleteModel(name));
}
async function deleteModel(name) {
  try { await apiFetch(`/models/${name}`,{method:'DELETE'}); toast(`Deleted "${name}"`,'success'); loadModels(); } catch {}
}
async function showModelInfo(name) {
  try {
    const info = await apiFetch(`/models/${name}/info`);
    openModal(`Model info: ${name}`,
      `<pre style="max-height:420px;overflow:auto;background:var(--bg3);padding:10px;border-radius:var(--radius);font-size:11px;">${escHtml(JSON.stringify(info, null, 2))}</pre>`,
      () => {});
    document.getElementById('modal-confirm-btn').style.display = 'none';
  } catch {}
}
async function reloadModel(name) {
  try {
    const r = await apiFetch(`/models/${name}/reload`, {method:'POST'});
    toast(`${name}: ${r.status || 'reloaded'}`, 'success');
    loadModels();
  } catch {}
}
async function refreshModel(name) {
  try {
    await apiFetch(`/models/${name}/refresh`, {method:'POST'});
    toast(`${name}: metadata refreshed`, 'success');
    loadModels();
  } catch {}
}
function confirmDeleteEnsemble(name) {
  openModal(`Delete ensemble "${name}"?`,'',() => deleteEnsemble(name));
}
async function deleteEnsemble(name) {
  try { await apiFetch(`/ensemble/${name}`,{method:'DELETE'}); toast(`Deleted "${name}"`,'success'); loadModels(); } catch {}
}
async function validateEnsemble(name) {
  try {
    const v = await apiFetch(`/ensemble/${name}/validate`);
    openModal(`Validate ensemble: ${name}`,
      `<pre style="max-height:420px;overflow:auto;background:var(--bg3);padding:10px;border-radius:var(--radius);font-size:11px;">${escHtml(JSON.stringify(v, null, 2))}</pre>`,
      () => {});
    document.getElementById('modal-confirm-btn').style.display = 'none';
    if (v.valid) toast(`${name}: valid`, 'success');
    else toast(`${name}: invalid`, 'error');
    loadModels();
  } catch {}
}

/* ══════════════════════════════════════════════════════════════
   DETECT
══════════════════════════════════════════════════════════════ */
function detectWithModel(name) {
  switchPage('detect');
  setTimeout(() => {
    selectAllModels(false);
    const cb = document.querySelector(`input[name="d-model-cb"][value="${name}"]`);
    if (cb) { cb.checked = true; _updateModelCount(); }
    else {
      loadModels().then(() => {
        const cb2 = document.querySelector(`input[name="d-model-cb"][value="${name}"]`);
        if (cb2) { cb2.checked = true; _updateModelCount(); }
      });
    }
  }, 80);
}
function handleDragOver(e) { e.preventDefault(); document.getElementById('dropzone').classList.add('drag'); }
document.addEventListener('dragleave', () => document.getElementById('dropzone').classList.remove('drag'));

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) loadImageFile(f);
}
function handleFileSelect(e) { if (e.target.files[0]) loadImageFile(e.target.files[0]); }

function loadImageFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    const wrap = document.getElementById('canvas-wrap');
    const pc = document.getElementById('preview-canvas');
    const oc = document.getElementById('overlay-canvas');
    const maxW = wrap.parentElement.clientWidth - 28;
    const maxH = 720;
    const scale = Math.min(maxW / img.width, maxH / img.height);
    pc.width  = Math.round(img.width  * scale);
    pc.height = Math.round(img.height * scale);
    oc.width = pc.width; oc.height = pc.height;
    oc.style.width = pc.width + 'px'; oc.style.height = pc.height + 'px';
    pc.getContext('2d').drawImage(img, 0, 0, pc.width, pc.height);
    document.getElementById('dropzone').style.display = 'none';
    wrap.style.display = 'block';
    document.getElementById('clear-btn').style.display = '';
    document.getElementById('run-btn').disabled = false;
    clearResults();
  };
  img.src = url;
}

function clearDetect() {
  currentFile = null; currentImage = null;
  document.getElementById('dropzone').style.display = '';
  document.getElementById('canvas-wrap').style.display = 'none';
  document.getElementById('clear-btn').style.display = 'none';
  document.getElementById('run-btn').disabled = true;
  clearResults();
}
function clearResults() {
  document.getElementById('results-list').innerHTML = '<div class="empty-state"><div class="empty-icon">⬤</div>No results yet</div>';
  document.getElementById('det-count').textContent = '';
  document.getElementById('detect-stats').innerHTML = '';
  const oc = document.getElementById('overlay-canvas');
  if (oc) oc.getContext('2d').clearRect(0, 0, oc.width, oc.height);
}

async function runDetect() {
  const models = getSelectedModels();
  if (!models.length) { toast(currentLanguage === 'vi' ? 'Chọn ít nhất 1 mô hình' : 'Select at least 1 model', 'error'); return; }
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spin">⟳</span> Running…';
  clearResults();

  let fileToSend;
  if (detectMode === 'video') {
    const vid = document.getElementById('detect-video');
    if (!vid.src || vid.videoWidth === 0) {
      toast('Load a video file first','error');
      btn.disabled = false; btn.textContent = '▶ RUN INFERENCE'; return;
    }
    fileToSend = await new Promise(res => {
      const cap = document.createElement('canvas');
      cap.width = vid.videoWidth; cap.height = vid.videoHeight;
      cap.getContext('2d').drawImage(vid, 0, 0);
      cap.toBlob(b => res(new File([b], 'frame.jpg', {type:'image/jpeg'})), 'image/jpeg', 0.92);
    });
  } else {
    if (!currentFile) { btn.disabled = false; btn.textContent = '▶ RUN INFERENCE'; return; }
    fileToSend = currentFile;
  }

  const cls    = document.getElementById('d-classes').value.trim();
  const imgsz  = document.getElementById('d-imgsz').value || '640';
  const conf   = document.getElementById('d-conf').value || '0.5';
  const iou    = document.getElementById('d-iou').value || '0.45';
  HOST = document.getElementById('host-input').value.replace(/\/$/, '');
  const t0 = performance.now();

  const promptModels = missingPromptModels(models, cls);
  if (promptModels.length) {
    toast(`"${promptModels[0]}" requires YOLOE prompts (e.g. person,car)`, 'error');
    btn.disabled = false; btn.textContent = '▶ RUN INFERENCE'; return;
  }

  try {
    // Run all models in parallel
    const results = await Promise.all(models.map(async model => {
      const form = new FormData();
      form.append('file', fileToSend);
      form.append('model', model);
      appendPromptFormIfNeeded(form, model, cls);
      form.append('imgsz', imgsz);
      form.append('conf', conf);
      form.append('iou', iou);
      const r = await fetch(HOST + '/detect', { method:'POST', body:form });
      if (!r.ok) { const e = await r.json(); throw new Error(`[${model}] ${e.detail||r.statusText}`); }
      const data = await r.json();
      data.annotations = normalizeAnnotations(data.annotations, model);
      return data;
    }));

    // Merge annotations from all models
    const merged = {
      annotations: results.flatMap(r => r.annotations || []),
      image_shape: results[0]?.image_shape,
      inference_imgsz: results[0]?.inference_imgsz,
      timing_ms: results.length === 1 ? results[0]?.timing_ms : null,
      per_model_timing_ms: Object.fromEntries(results.map((r, i) => [models[i], r.timing_ms || null])),
    };
    const ms = (performance.now() - t0).toFixed(0);
    if (detectMode === 'video') renderVideoDetections(merged, ms);
    else renderDetections(merged, ms);
  } catch (e) { toast('Detect error: ' + e.message, 'error'); }
  btn.disabled = false; btn.textContent = '▶ RUN INFERENCE';
}

function renderDetections(data, ms) {
  const anns = data.annotations || [];
  document.getElementById('det-count').textContent = `${anns.length} detection${anns.length!==1?'s':''}`;
  document.getElementById('detect-stats').innerHTML = timingStatsHtml(data, { totalMs: ms });
  if (!anns.length) {
    document.getElementById('results-list').innerHTML = '<div class="empty-state"><div class="empty-icon">⬤</div>No detections</div>';
    return;
  }
  const catMap = buildCatMap(anns);
  document.getElementById('results-list').innerHTML = anns.map(a => `
    <div class="result-item" style="border-left-color:${catMap[annColorKey(a)]}">
      <span class="result-cat" style="color:${catMap[annColorKey(a)]}">${escHtml(annLabel(a))}</span>
      <span class="result-score">${(a.score*100).toFixed(1)}%</span>
      <span class="result-bbox">[${a.bbox.map(v=>v.toFixed(0)).join(', ')}]</span>
      ${a.source_model ? `<span class="source-badge">${escHtml(a.source_model)}</span>` : ''}
      ${a.segmentation ? `<span class="seg-badge">seg</span>` : ''}
    </div>`).join('');

  const oc = document.getElementById('overlay-canvas');
  const ctx = oc.getContext('2d');
  ctx.clearRect(0, 0, oc.width, oc.height);
  // Draw on the overlay canvas sized to match the preview canvas display
  drawAnnotations(ctx, oc.width, oc.height, anns, data.image_shape, catMap);
}

/* ══════════════════════════════════════════════════════════════
   DETECT MODE / VIDEO / WEBCAM
══════════════════════════════════════════════════════════════ */
function setDetectMode(mode) {
  detectMode = mode;
  document.querySelectorAll('.detect-mode-btn').forEach((b, i) =>
    b.classList.toggle('active', ['image','video','webcam'][i] === mode));
  document.getElementById('detect-image-pane').style.display  = mode === 'image'  ? '' : 'none';
  document.getElementById('detect-video-pane').style.display  = mode === 'video'  ? 'block' : 'none';
  document.getElementById('detect-webcam-pane').style.display = mode === 'webcam' ? 'block' : 'none';
  if (mode !== 'webcam') stopDetectWebcam();
  if (mode !== 'video')  stopVideoDetect();
  clearResults();
  const btn = document.getElementById('run-btn');
  btn.style.display = mode === 'webcam' ? 'none' : '';
  if (mode === 'image')  btn.disabled = !currentFile;
  if (mode === 'video')  btn.disabled = !document.getElementById('detect-video')._objUrl;
  if (mode === 'webcam') enumerateDetectCameras();
}

function loadVidFile(e) {
  const f = e.target.files[0]; if (!f) return;
  const vid = document.getElementById('detect-video');
  if (vid._objUrl) URL.revokeObjectURL(vid._objUrl);
  vid._objUrl = URL.createObjectURL(f);
  vid.src = vid._objUrl;
  vid.load();
  vid.ontimeupdate = _updateVidTime;
  vid.onended = () => {
    const playBtn = document.getElementById('vid-playpause');
    if (playBtn) playBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:11px;height:11px;display:block;margin:auto;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
  };
  vid.onloadedmetadata = () => {
    document.getElementById('vid-seek').max = vid.duration;
    _updateVidTime();
    const ov = document.getElementById('detect-video-overlay');
    ov.width = vid.videoWidth; ov.height = vid.videoHeight;
  };
  document.getElementById('vid-dropzone').style.display = 'none';
  document.getElementById('vid-player-wrap').style.display = '';
  document.getElementById('run-btn').disabled = false;
  document.getElementById('vid-det-start').disabled = false;
  document.getElementById('vs1').classList.add('spd-active');
  ['vs2','vs3','vs4'].forEach(id => document.getElementById(id).classList.remove('spd-active'));
}

function _updateVidTime() {
  const vid = document.getElementById('detect-video');
  document.getElementById('vid-seek').value = vid.currentTime;
  document.getElementById('vid-time').textContent = _fmtT(vid.currentTime) + ' / ' + _fmtT(vid.duration || 0);
}

function vidPlayPause() {
  const vid = document.getElementById('detect-video'), btn = document.getElementById('vid-playpause');
  const playSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:11px;height:11px;display:block;margin:auto;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
  const pauseSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:11px;height:11px;display:block;margin:auto;"><line x1="6" y1="4" x2="6" y2="20"></line><line x1="18" y1="4" x2="18" y2="20"></line></svg>`;
  if (vid.paused) { 
    vid.play(); 
    if (btn) btn.innerHTML = pauseSvg; 
  } else { 
    vid.pause(); 
    if (btn) btn.innerHTML = playSvg; 
  }
}

function vidSeek(val) {
  document.getElementById('detect-video').currentTime = parseFloat(val);
  const ov = document.getElementById('detect-video-overlay');
  ov.getContext('2d').clearRect(0, 0, ov.width, ov.height);
}

function vidSpeed(s, el) {
  document.getElementById('detect-video').playbackRate = s;
  ['vs1','vs2','vs3','vs4'].forEach(id => document.getElementById(id).classList.remove('spd-active'));
  if (el) el.classList.add('spd-active');
}

function vidFullscreen() {
  const fsEl = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullscreenElement || document.msFullscreenElement;
  if (fsEl) {
    const exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
    if (exit) exit.call(document);
    return;
  }
  const el = document.getElementById('vid-player-wrap');
  const fn = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  if (!fn) return;
  fn.call(el);
  const onFsChange = () => {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullscreenElement || document.msFullscreenElement;
    el.classList.toggle('fs-active', !!fsEl);
    requestAnimationFrame(syncVideoOverlay);
    setTimeout(syncVideoOverlay, 120);
    if (!fsEl) {
      document.removeEventListener('fullscreenchange', onFsChange);
      document.removeEventListener('webkitfullscreenchange', onFsChange);
      document.removeEventListener('mozfullscreenchange', onFsChange);
    }
  };
  document.addEventListener('fullscreenchange', onFsChange);
  document.addEventListener('webkitfullscreenchange', onFsChange);
  document.addEventListener('mozfullscreenchange', onFsChange);
}

function syncVideoOverlay() {
  const vid = document.getElementById('detect-video');
  const ov = document.getElementById('detect-video-overlay');
  const wrap = document.getElementById('vid-player-wrap');
  if (!vid || !ov || !wrap) return { w: vid?.videoWidth || 640, h: vid?.videoHeight || 480 };
  const w = vid.videoWidth || 640;
  const h = vid.videoHeight || 480;
  if (ov.width !== w) ov.width = w;
  if (ov.height !== h) ov.height = h;

  const vr = vid.getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  ov.style.left = (vr.left - wr.left) + 'px';
  ov.style.top = (vr.top - wr.top) + 'px';
  ov.style.width = vr.width + 'px';
  ov.style.height = vr.height + 'px';
  return { w, h };
}

function _fmtT(s) {
  if (!s || isNaN(s)) return '0:00';
  return Math.floor(s/60) + ':' + String(Math.floor(s%60)).padStart(2,'0');
}

function renderVideoDetections(data, ms) {
  const anns = data.annotations || [];
  document.getElementById('det-count').textContent = `${anns.length} detection${anns.length!==1?'s':''}`;
  document.getElementById('detect-stats').innerHTML = timingStatsHtml(data, { totalMs: ms });
  const vid = document.getElementById('detect-video');
  const ov  = document.getElementById('detect-video-overlay');
  const { w: vw, h: vh } = syncVideoOverlay();
  const ctx = ov.getContext('2d');
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!anns.length) {
    document.getElementById('results-list').innerHTML = '<div class="empty-state"><div class="empty-icon">⬤</div>No detections</div>';
    return;
  }
  const catMap = buildCatMap(anns);
  drawAnnotations(ctx, vw, vh, anns, data.image_shape, catMap);
  document.getElementById('results-list').innerHTML = anns.map(a => `
    <div class="result-item" style="border-left-color:${catMap[annColorKey(a)]}">
      <span class="result-cat" style="color:${catMap[annColorKey(a)]}">${escHtml(annLabel(a))}</span>
      <span class="result-score">${(a.score*100).toFixed(1)}%</span>
      <span class="result-bbox">[${a.bbox.map(v=>v.toFixed(0)).join(', ')}]</span>
      ${a.source_model?`<span class="source-badge">${escHtml(a.source_model)}</span>`:''}
      ${a.segmentation?`<span class="seg-badge">seg</span>`:''}
    </div>`).join('');
}

async function startDetectWebcam() {
  const models = getSelectedModels();
  if (!models.length) { toast(currentLanguage === 'vi' ? 'Chọn ít nhất 1 mô hình' : 'Select at least 1 model', 'error'); return; }
  const classes = document.getElementById('d-classes').value.trim();
  const promptModels = missingPromptModels(models, classes);
  if (promptModels.length) { toast(`"${promptModels[0]}" requires YOLOE prompts`, 'error'); return; }
  stopDetectWebcam();
  resetDetectLivePerf('webcam');
  detectWebcamExactFrame = null;
  detectWebcamExactFrameAt = 0;
  const wcGen = ++detectWebcamGeneration;
  const deviceId = document.getElementById('webcam-det-device').value;
  try {
    const constraint = deviceId === '__env__' ? { facingMode: { exact: 'environment' } }
                     : deviceId === '__user__' ? { facingMode: { exact: 'user' } }
                     : deviceId ? { deviceId: { exact: deviceId } }
                     : true;
    detectWebcamStream = await navigator.mediaDevices.getUserMedia({ video: constraint, audio: false });
    detectWebcamStream.getTracks().forEach(track => {
      track.onended = () => {
        if (wcGen === detectWebcamGeneration) stopDetectWebcam();
      };
    });
    detectWebcamVideo = document.createElement('video');
    detectWebcamVideo.srcObject = detectWebcamStream;
    detectWebcamVideo.muted = true; detectWebcamVideo.playsInline = true;
    detectWebcamVideo.play().catch(() => {});

    HOST = document.getElementById('host-input').value.replace(/\/$/, '');
    const wsHost  = HOST.replace(/^http/,'ws');
    const imgsz   = document.getElementById('d-imgsz').value || '640';
    const conf    = document.getElementById('d-conf').value  || '0.5';
    const liveFps = getDetectLiveFps(10);
    const webcamTracking = document.getElementById('detect-global-tracking')?.checked || false;

    const _annsByModel = {};
    const _wsList = models.map(model => {
      let url = `${wsHost}/ws/stream?model=${encodeURIComponent(model)}&fps=${liveFps}&imgsz=${imgsz}&conf=${conf}`;
      if (webcamTracking) url += '&track=true';
      url = appendPromptIfNeeded(url, model, classes);
      const ws = new WebSocket(url);
      ws._model = model;
      ws._inflight = 0;  // frames awaiting response; max 2 in-flight for pipelining
      ws._sentAt = 0;
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        if (wcGen === detectWebcamGeneration) toast(`Webcam inference ready: ${model}`, 'success', 1600);
      };
      ws.onmessage = ev => {
        const clientMs = ws._sentAt ? performance.now() - ws._sentAt : null;
        ws._inflight = Math.max(0, ws._inflight - 1);
        if (wcGen !== detectWebcamGeneration || !detectWebcamStream) return;
        try {
          const data = JSON.parse(ev.data);
          if (data.dropped) return;
          updateDetectLiveStats(data, {
            clientMs,
            requestedFps: liveFps,
            resultFps: noteDetectLiveResult('webcam'),
          });
          const anns = normalizeAnnotations(data.annotations, model);
          _annsByModel[model] = { anns, imgShape: data.image_shape };
          detectWebcamAnns = Object.values(_annsByModel).flatMap(v => v.anns);
          detectWebcamImgShape = data.image_shape;
          detectWebcamAnns.forEach(a => {
            const key = annColorKey(a);
            if (!detectWebcamCatMap[key])
              detectWebcamCatMap[key] = COLORS[Object.keys(detectWebcamCatMap).length % COLORS.length];
          });
          document.getElementById('det-count').textContent = detectWebcamAnns.length + ' det';
          document.getElementById('results-list').innerHTML = detectWebcamAnns.length
            ? detectWebcamAnns.map(a=>`
                <div class="result-item" style="border-left-color:${detectWebcamCatMap[annColorKey(a)]}">
                  <span class="result-cat" style="color:${detectWebcamCatMap[annColorKey(a)]}">${escHtml(annLabel(a))}</span>
                  <span class="result-score">${(a.score*100).toFixed(1)}%</span>
                  ${a.source_model?`<span class="source-badge">${escHtml(a.source_model)}</span>`:''}
                </div>`).join('')
            : '<div class="empty-state"><div class="empty-icon">⬤</div>No detections</div>';

        } catch {}
      };
      ws.onerror = () => {
        ws._inflight = 0;
        if (wcGen === detectWebcamGeneration) toast(`Webcam WS error: ${model}`, 'error', 5000);
      };
      ws.onclose = e => {
        ws._inflight = 0;
        if (wcGen === detectWebcamGeneration && detectWebcamStream && e.code !== 1000) {
          toast(`Webcam WS closed: ${model} (${e.code || 1005})`, 'error', 6000);
        }
      };
      return ws;
    });
    detectWebcamWs = {
      _wsList,
      readyState: 1,
      close() { this._wsList.forEach(w => w.close()); },
      send(ab)  {
        const now = performance.now();
        let sent = 0;
        this._wsList.forEach(w => {
          if (w.readyState === 1 && w._inflight < 2) {  // pipeline: up to 2 frames in-flight
            w._inflight++;
            w._sentAt = now;
            w._pendingFrame = ab;
            w.send(ab);
            sent++;
          }
        });
        return sent;
      },
      hasReadyToSend() {
        return this._wsList.some(w => w.readyState === 1 && w._inflight < 2);
      }
    };

    document.getElementById('webcam-det-start').style.display = 'none';
    document.getElementById('webcam-det-stop').style.display  = '';

    const canvas = document.getElementById('webcam-detect-canvas');
    const cap = document.createElement('canvas');
    let lastSend = 0;
    const _wcInterval = 1000 / liveFps;
    const loop = ts => {
      if (!detectWebcamStream) return;
      detectWebcamLoopId = requestAnimationFrame(loop);
      if (!detectWebcamVideo || detectWebcamVideo.readyState < 2) return;
      const w = detectWebcamVideo.videoWidth || 640, h = detectWebcamVideo.videoHeight || 480;
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      const ctx = canvas.getContext('2d');
      // Always draw live webcam at full rAF rate for smooth playback
      ctx.drawImage(detectWebcamVideo, 0, 0, w, h);
      // Overlay bounding boxes from latest inference result.
      // In exact mode, detectWebcamImgShape is from the inference frame sent;
      // drawAnnotations remaps coordinates to current canvas size automatically.
      if (detectWebcamAnns.length && detectWebcamImgShape) {
        drawAnnotations(ctx, w, h, detectWebcamAnns, detectWebcamImgShape, detectWebcamCatMap);
      }
      // Step 3: send a clean frame to inference at throttled rate
      if (ts - lastSend >= _wcInterval) {
        if (!detectWebcamWs?.hasReadyToSend()) return;
        lastSend = ts;
        // Cap send resolution at 1280px — server letterboxes to imgsz anyway; halves JPEG payload for HD cams
        const MAX_SEND_PX = 1280;
        const capScale = Math.min(1, MAX_SEND_PX / w);
        const sendW = Math.round(w * capScale);
        const sendH = Math.round(h * capScale);
        if (cap.width !== sendW) cap.width = sendW;
        if (cap.height !== sendH) cap.height = sendH;
        cap.getContext('2d').drawImage(detectWebcamVideo, 0, 0, sendW, sendH);
        cap.toBlob(blob => {
          if (!blob || wcGen !== detectWebcamGeneration || !detectWebcamStream) return;
          blob.arrayBuffer().then(ab => {
            if (wcGen === detectWebcamGeneration && detectWebcamStream) detectWebcamWs?.send(ab);
          });
        }, 'image/jpeg', 0.72);  // 0.72 vs 0.8: ~25% smaller, same inference quality
      }
    };
    detectWebcamLoopId = requestAnimationFrame(loop);
    toast(`Webcam started — ${models.length} model${models.length>1?'s':''}`, 'success');
  } catch(e) {
    toast('Webcam error: ' + e.message,'error');
    stopDetectWebcam();
  }
}


function startVideoDetect() {
  const models = getSelectedModels();
  if (!models.length) { toast(currentLanguage === 'vi' ? 'Chọn ít nhất 1 mô hình' : 'Select at least 1 model', 'error'); return; }
  const classes = document.getElementById('d-classes').value.trim();
  const promptModels = missingPromptModels(models, classes);
  if (promptModels.length) { toast(`"${promptModels[0]}" requires YOLOE prompts`, 'error'); return; }
  const vid = document.getElementById('detect-video');
  if (!vid.src || vid.videoWidth === 0) { toast('Load a video file first','error'); return; }

  // WebSocket path (default)
  stopVideoDetect();
  resetDetectLivePerf('video');
  videoDetectCatMap = {};
  videoDetectExactFrame = null;
  videoDetectExactFrameAt = 0;
  HOST = document.getElementById('host-input').value.replace(/\/$/, '');
  const wsHost = HOST.replace(/^http/,'ws');
  const imgsz   = document.getElementById('d-imgsz').value || '640';
  const conf    = document.getElementById('d-conf').value  || '0.5';
  const liveFps = getDetectLiveFps(10);
  const detectTracking = document.getElementById('detect-global-tracking')?.checked || false;
  const _vidAnnsByModel = {};
  const _vidWsList = models.map(model => {
    let url = `${wsHost}/ws/stream?model=${encodeURIComponent(model)}&fps=${liveFps}&imgsz=${imgsz}&conf=${conf}`;
    if (detectTracking) url += '&track=true';
    url = appendPromptIfNeeded(url, model, classes);
    const ws = new WebSocket(url);
    ws._model = model;
    ws._inflight = 0;  // pipeline: max 2 frames in-flight
    ws._sentAt = 0;
    ws.binaryType = 'arraybuffer';
    ws.onmessage = ev => {
      const clientMs = ws._sentAt ? performance.now() - ws._sentAt : null;
      ws._inflight = Math.max(0, ws._inflight - 1);
      try {
        const data = JSON.parse(ev.data);
        if (data.dropped) return;
        updateDetectLiveStats(data, {
          clientMs,
          requestedFps: liveFps,
          resultFps: noteDetectLiveResult('video'),
        });
        const anns = normalizeAnnotations(data.annotations, model);
        _vidAnnsByModel[model] = { anns, imgShape: data.image_shape };
        videoDetectAnns = Object.values(_vidAnnsByModel).flatMap(v => v.anns);
        videoDetectImgShape = data.image_shape;
        videoDetectAnns.forEach(a => {
          const key = annColorKey(a);
          if (!videoDetectCatMap[key])
            videoDetectCatMap[key] = COLORS[Object.keys(videoDetectCatMap).length % COLORS.length];
        });
        document.getElementById('det-count').textContent = videoDetectAnns.length + ' det';
        document.getElementById('results-list').innerHTML = videoDetectAnns.length
          ? videoDetectAnns.map(a => `
              <div class="result-item" style="border-left-color:${videoDetectCatMap[annColorKey(a)]}">
                <span class="result-cat" style="color:${videoDetectCatMap[annColorKey(a)]}">${escHtml(annLabel(a))}</span>
                <span class="result-score">${(a.score*100).toFixed(1)}%</span>
                <span class="result-bbox">[${a.bbox.map(v=>v.toFixed(0)).join(', ')}]</span>
                ${a.source_model?`<span class="source-badge">${escHtml(a.source_model)}</span>`:''}
                ${a.segmentation?`<span class="seg-badge">seg</span>`:''}
              </div>`).join('')
          : '<div class="empty-state"><div class="empty-icon">⬤</div>No detections</div>';

      } catch {}
    };
    ws.onerror = () => { ws._inflight = 0; };
    ws.onclose = () => { ws._inflight = 0; };
    return ws;
  });
  videoDetectWs = {
    _vidWsList,
    readyState: 1,
    close() { this._vidWsList.forEach(w => w.close()); },
    send(ab)  {
      const now = performance.now();
      this._vidWsList.forEach(w => {
        if (w.readyState === 1 && w._inflight < 2) {
          w._inflight++;
          w._sentAt = now;
          w._pendingFrame = ab;
          w.send(ab);
        }
      });
    },
    hasReadyToSend() {
      return this._vidWsList.some(w => w.readyState === 1 && w._inflight < 2);
    }
  };
  document.getElementById('vid-det-start').style.display = 'none';
  document.getElementById('vid-det-stop').style.display  = '';
  if (vid.ended) vid.currentTime = 0;
  vid.play().then(() => {
    const playBtn = document.getElementById('vid-playpause');
    if (playBtn) playBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:11px;height:11px;display:block;margin:auto;"><line x1="6" y1="4" x2="6" y2="20"></line><line x1="18" y1="4" x2="18" y2="20"></line></svg>`;
  }).catch(() => {});

  const ov = document.getElementById('detect-video-overlay');
  const cap = document.createElement('canvas');
  let lastSend = 0;
  const _vdInterval = 1000 / liveFps;
  const loop = ts => {
    if (!videoDetectWs) return;
    videoDetectLoopId = requestAnimationFrame(loop);
    if (vid.readyState < 2 || vid.paused) return;
    const w = vid.videoWidth || 640, h = vid.videoHeight || 480;
    syncVideoOverlay();
    const octx = ov.getContext('2d');
    octx.clearRect(0, 0, ov.width, ov.height);
    // Always draw live video at native rAF rate for smooth playback
    octx.clearRect(0, 0, ov.width, ov.height);
    // Overlay bounding boxes from latest inference result.
    // In exact mode, videoDetectImgShape is the inference-frame coordinate space;
    // drawAnnotations remaps to current overlay canvas size automatically.
    if (videoDetectAnns.length && videoDetectImgShape) {
      drawAnnotations(octx, ov.width || w, ov.height || h, videoDetectAnns, videoDetectImgShape, videoDetectCatMap);
    }
    // Send clean frame at throttled rate
    if (ts - lastSend >= _vdInterval) {
      if (!videoDetectWs?.hasReadyToSend()) return;
      lastSend = ts;
      if (cap.width !== w) cap.width = w;
      if (cap.height !== h) cap.height = h;
      cap.getContext('2d').drawImage(vid, 0, 0, w, h);
      cap.toBlob(blob => {
        blob.arrayBuffer().then(ab => videoDetectWs?.send(ab));
      }, 'image/jpeg', 0.8);
    }
  };
  videoDetectLoopId = requestAnimationFrame(loop);
  toast(`Video detect started — ${models.length} model${models.length>1?'s':''}`, 'success');
}

function stopVideoDetect() {
  if (videoDetectLoopId) { cancelAnimationFrame(videoDetectLoopId); videoDetectLoopId = null; }
  if (videoDetectWs)     { videoDetectWs.close(); videoDetectWs = null; }
  videoDetectExactFrame = null;
  videoDetectExactFrameAt = 0;
  videoDetectAnns = []; videoDetectImgShape = null;
  const s = document.getElementById('vid-det-start'), e = document.getElementById('vid-det-stop');
  if (s) { s.style.display = ''; }
  if (e) { e.style.display = 'none'; }
  const vid = document.getElementById('detect-video');
  if (vid && !vid.paused) {
    vid.pause();
    const playBtn = document.getElementById('vid-playpause');
    if (playBtn) playBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:11px;height:11px;display:block;margin:auto;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
  }
  if (!vid || !vid._objUrl) {
    const dz = document.getElementById('vid-dropzone');
    const pw = document.getElementById('vid-player-wrap');
    if (dz) dz.style.display = '';
    if (pw) pw.style.display = 'none';
  }
}

function webcamDetFullscreen() {
  const fsEl = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullscreenElement || document.msFullscreenElement;
  if (fsEl) {
    const exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
    if (exit) exit.call(document);
    return;
  }
  const el = document.getElementById('webcam-detect-wrap');
  const fn = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  if (fn) fn.call(el);
}

function tileFullscreen(id) {
  const wrap = document.getElementById(`canvas-${id}`)?.parentElement;
  if (!wrap) return;
  const fsEl = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement;
  if (fsEl) {
    const exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
    if (exit) exit.call(document);
    return;
  }
  const fn = wrap.requestFullscreen || wrap.webkitRequestFullscreen || wrap.mozRequestFullScreen || wrap.msRequestFullscreen;
  if (fn) fn.call(wrap);
}

function stopDetectWebcam() {
  detectWebcamGeneration++;
  if (detectWebcamLoopId) { cancelAnimationFrame(detectWebcamLoopId); detectWebcamLoopId = null; }
  if (detectWebcamWs)     { detectWebcamWs.close(); detectWebcamWs = null; }
  if (detectWebcamStream) { detectWebcamStream.getTracks().forEach(t => t.stop()); detectWebcamStream = null; }
  detectWebcamVideo = null; detectWebcamAnns = []; detectWebcamImgShape = null; detectWebcamCatMap = {};
  detectWebcamExactFrame = null; detectWebcamExactFrameAt = 0;
  const s = document.getElementById('webcam-det-start'), e = document.getElementById('webcam-det-stop');
  if (s) s.style.display = ''; if (e) e.style.display = 'none';
}

/* ══════════════════════════════════════════════════════════════
   RLE SEGMENTATION DECODER
══════════════════════════════════════════════════════════════ */
function decodeCOCOrle(rle) {
  // Returns flat Uint8Array (row-major) [imgH x imgW]
  const [imgH, imgW] = rle.size;
  const s = rle.counts;
  const cnts = [];
  if (typeof s === 'string') {
    let p = 0;
    while (p < s.length) {
      let x = 0, k = 0, more = 1;
      while (more) {
        const c = s.charCodeAt(p) - 48;
        x |= (c & 0x1f) << (5 * k);
        more = c & 0x20;
        p++; k++;
        if (!more && (c & 0x10)) x |= (-1 << (5 * k));
      }
      // pycocotools _mask.c: if(m>2) x += cnts[m-2]  (delta from two positions back)
      if (cnts.length > 2) x += cnts[cnts.length - 2];
      cnts.push(x);
    }
  } else if (Array.isArray(s)) {
    cnts.push(...s);
  }
  const mask = new Uint8Array(imgH * imgW);
  let pos = 0, fill = 0;
  for (const cnt of cnts) {
    if (fill) {
      for (let j = 0; j < cnt; j++) {
        const col = Math.floor(pos / imgH);
        const row = pos % imgH;
        if (col < imgW) mask[row * imgW + col] = 1;
        pos++;
      }
    } else {
      pos += cnt;
    }
    fill ^= 1;
  }
  return { mask, imgH, imgW };
}

// getMaskCanvas removed — masks are now rendered at display resolution in drawAnnotations
// to avoid Android WebView max-canvas-size limits that caused silent blank draws.

/* ══════════════════════════════════════════════════════════════
   DRAW ANNOTATIONS (single canvas — video frame + overlay)
══════════════════════════════════════════════════════════════ */
function buildCatMap(anns) {
  const m = {};
  anns.forEach(a => {
    const key = annColorKey(a);
    if (!m[key]) m[key] = COLORS[Object.keys(m).length % COLORS.length];
  });
  return m;
}

function renderTileDetectionList(instance) {
  const detsEl = document.getElementById(`tile-dets-${instance.id}`);
  const cntEl = document.getElementById(`tile-det-cnt-${instance.id}`);
  const anns = instance.lastAnns || [];
  if (cntEl) cntEl.textContent = anns.length ? `(${anns.length})` : '';
  if (!detsEl) return;
  if (!anns.length) {
    detsEl.innerHTML = `<div class="tile-no-det">${uiLabel('No detections')}</div>`;
    return;
  }
  const groups = new Map();
  anns.forEach(a => {
    const label = annLabel(a);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(a);
  });
  if (!instance.expandedDetectionGroups) instance.expandedDetectionGroups = new Set();
  detsEl.innerHTML = [...groups.entries()].map(([label, items], gi) => {
    const top = Math.max(...items.map(a => Number(a.score) || 0));
    const key = `detg-${instance.id}-${safeDomId(label)}`;
    const expanded = instance.expandedDetectionGroups.has(label);
    const color = instance.catColorMap[annColorKey(items[0])] || COLORS[gi % COLORS.length];
    const details = items.map((a, i) => `
      <div class="tile-det-row" style="border-left-color:${instance.catColorMap[annColorKey(a)] || color}">
        <span class="tile-det-cat" style="color:${instance.catColorMap[annColorKey(a)] || color}">${escHtml(label)} #${i + 1}</span>
        ${a.source_model ? `<span class="source-badge">${escHtml(a.source_model)}</span>` : ''}
        <span class="tile-det-sc">${((a.score || 0) * 100).toFixed(1)}%</span>
      </div>`).join('');
    return `
      <div class="tile-det-group" style="border-left-color:${color}">
        <button class="tile-det-summary" onclick="toggleDetectionGroup(event, '${instance.id}', '${escAttr(label)}')">
          <span class="tile-det-cat" style="color:${color}">${escHtml(label)}</span>
          <span class="tile-det-count">${items.length}x</span>
          <span class="tile-det-sc">top ${(top * 100).toFixed(0)}%</span>
        </button>
        <div class="tile-det-details ${expanded ? '' : 'collapsed'}" id="${key}">${details}</div>
      </div>`;
  }).join('');
}

function safeDomId(s) {
  return String(s).replace(/[^a-zA-Z0-9_-]+/g, '_');
}

function escAttr(s) {
  return escHtml(s).replace(/'/g, '&#39;');
}

function jsAttr(s) {
  return escHtml(String(s)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\r/g, '\\r')
    .replace(/\n/g, '\\n'));
}

function modelIcon(name) {
  const icons = {
    play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg>',
    edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20h9"></path><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"></path></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"></path></svg>',
    trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="M6 6l1 15h10l1-15"></path></svg>',
    info: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 10v7"></path><path d="M12 7h.01"></path></svg>',
    refresh: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12a9 9 0 0 1-15.5 6.2"></path><path d="M3 12A9 9 0 0 1 18.5 5.8"></path><path d="M18 2v4h-4"></path><path d="M6 22v-4h4"></path></svg>',
    reload: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 0 1 15-6.7"></path><path d="M18 3v5h-5"></path><path d="M21 12a9 9 0 0 1-15 6.7"></path><path d="M6 21v-5h5"></path></svg>',
    dots: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="5" r="1.5"></circle><circle cx="12" cy="12" r="1.5"></circle><circle cx="12" cy="19" r="1.5"></circle></svg>',
  };
  return icons[name] || '';
}

function closeModelMenus() {
  document.querySelectorAll('.model-menu.open').forEach(el => el.classList.remove('open'));
}

function toggleModelMenu(event, id) {
  event?.preventDefault?.();
  event?.stopPropagation?.();
  document.querySelectorAll('.model-menu.open').forEach(el => {
    if (el.id !== id) el.classList.remove('open');
  });
  document.getElementById(id)?.classList.toggle('open');
}

document.addEventListener('click', event => {
  if (!event.target.closest?.('.model-menu')) closeModelMenus();
});

function toggleDetectionGroup(event, streamId, label) {
  event?.preventDefault?.();
  event?.stopPropagation?.();
  const inst = streams.get(String(streamId));
  if (!inst) return;
  if (!inst.expandedDetectionGroups) inst.expandedDetectionGroups = new Set();
  if (inst.expandedDetectionGroups.has(label)) inst.expandedDetectionGroups.delete(label);
  else inst.expandedDetectionGroups.add(label);
  renderTileDetectionList(inst);
}

function drawAnnotations(ctx, canvasW, canvasH, anns, imgShape, catMap) {
  if (!imgShape || !anns.length) return;
  const [imgH, imgW] = imgShape;
  const sx = canvasW / imgW;
  const sy = canvasH / imgH;

  // Pass 1: segmentation masks — built at *display* resolution so the
  // intermediate canvas stays small and never hits Android's canvas size cap.
  const _masksToRender = anns.filter(a => a.segmentation?.counts);
  if (_masksToRender.length) {
    const mCanvas = document.createElement('canvas');
    mCanvas.width = canvasW; mCanvas.height = canvasH;
    const mCtx = mCanvas.getContext('2d');
    if (mCtx) {
      const scaleX = imgW / canvasW;
      const scaleY = imgH / canvasH;
      _masksToRender.forEach(a => {
        const color = catMap[annColorKey(a)] || COLORS[0];
        try {
          const cacheKey = maskCacheKey(a, canvasW, canvasH, color);
          let cached = a._maskCache;
          if (!cached || cached.key !== cacheKey) {
            const [r, g, b] = hexToRgb(color);
            const idata = mCtx.createImageData(canvasW, canvasH);
            const { mask, imgH: mH, imgW: mW } = decodeCOCOrle(a.segmentation);
            for (let py = 0; py < canvasH; py++) {
              for (let px = 0; px < canvasW; px++) {
                const mx = Math.min(Math.floor(px * scaleX), mW - 1);
                const my = Math.min(Math.floor(py * scaleY), mH - 1);
                if (mask[my * mW + mx]) {
                  const idx = (py * canvasW + px) * 4;
                  idata.data[idx]     = r;
                  idata.data[idx + 1] = g;
                  idata.data[idx + 2] = b;
                  idata.data[idx + 3] = 90;
                }
              }
            }
            const oneMask = document.createElement('canvas');
            oneMask.width = canvasW;
            oneMask.height = canvasH;
            oneMask.getContext('2d').putImageData(idata, 0, 0);
            cached = a._maskCache = { key: cacheKey, canvas: oneMask };
          }
          mCtx.drawImage(cached.canvas, 0, 0);
        } catch { /* skip malformed mask */ }
      });
      ctx.drawImage(mCanvas, 0, 0); // 1:1 draw — no scaling, safe on all platforms
    }
  }

  // Pass 2: bounding boxes + labels
  ctx.font = '700 16px "SFMono-Regular","SF Mono",ui-monospace,Menlo,Consolas,monospace';
  ctx.textBaseline = 'top';
  anns.forEach(a => {
    const [x, y, w, h] = a.bbox;
    const color = catMap[annColorKey(a)] || COLORS[0];
    const px = x * sx, py = y * sy, pw = w * sx, ph = h * sy;

    // Box — thicker stroke with a dark shadow outline for contrast
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.lineWidth = 5;
    ctx.strokeRect(px, py, pw, ph);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(px, py, pw, ph);

    // Label background + text
    const label = `${annLabel(a)} ${(a.score*100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 14;
    const bh = 27;
    const ly = py >= bh ? py - bh : py + ph;
    // Shadow backing for readability
    ctx.fillStyle = 'rgba(0,0,0,0.76)';
    ctx.fillRect(px - 1, ly - 1, tw + 2, bh + 2);
    ctx.fillStyle = color;
    ctx.fillRect(px, ly, tw, bh);
    ctx.fillStyle = '#030303';
    ctx.fillText(label, px + 7, ly + 5);
  });
}

/* ══════════════════════════════════════════════════════════════
   STREAM ENGINE
══════════════════════════════════════════════════════════════ */
class StreamInstance {
  constructor(cfg) {
    this.id    = cfg.id;
    this.name  = cfg.name;
    this.type  = cfg.type;   // 'webcam' | 'file' | 'hls' | 'ws' | 'server_rtsp'
    this.src   = cfg.src;    // URL string or File object
    this.models = (cfg.models && cfg.models.length) ? cfg.models : (cfg.model ? [cfg.model] : []);
    this.model = this.models[0] || '';
    this.classes = cfg.classes;
    this.imgsz = cfg.imgsz || 640;
    this.conf  = cfg.conf  || 0.5;
    this.fps   = clampFpsValue(cfg.fps || 30, 10);
    this.previewFps = clampFpsValue(cfg.previewFps || Math.min(this.fps, 10), 10);
    this.rtspBackend = normalizeRtspBackendChoice(cfg.rtspBackend || 'auto');
    this.overlayMode = normalizeOverlayMode(cfg.overlayMode || (cfg.syncRtspBoxes ? 'exact' : 'native_exact'));
    this.liveTransport = cfg.liveTransport || 'go2rtc';
    this.preferredLiveTransport = cfg.liveTransport || 'go2rtc';
    this.sourceMaxHeight = cfg.sourceMaxHeight == null ? 720 : Number(cfg.sourceMaxHeight);
    this.annotatedPreview = cfg.annotatedPreview ?? alignedBoxesModeEnabled(this.overlayMode);
    this.tab      = cfg.tab;
    this.deviceId = cfg.deviceId || null;
    this.enableTracking = cfg.enableTracking || false;
    this.enableRecording = cfg.enableRecording || false;

    this.active = false;
    this.inferWs   = null;
    this.inferWsList = [];
    this.srcWs     = null;
    this.eventWs   = null;
    this.previewWs = null;
    this.managedStreamId = null;
    this.go2rtcName = null;
    this.go2rtcPublicUrl = null;
    this.go2rtcPeer = null;
    this.go2rtcVideo = null;
    this.go2rtcFrameCb = null;
    this.go2rtcOverlayLoopId = null;
    this.srcWatchdogId = null;
    this.previewWatchdogId = null;
    this.previewReconnectTimer = null;
    this.lastSrcFrameAt = 0;
    this.srcReconnects = 0;
    this.srcReconnectDelayMs = 1000;
    this.srcStallTimeoutMs = 5000;
    this.srcAutoReconnect = true;
    this.videoEl   = null;
    this.mediaStream = null;
    this.loopId    = null;
    this.audioMuted = true;

    this.canvas    = null;
    this.catColorMap = {};
    this.lastAnns  = [];
    this.annsByModel = {};
    this.lastImgShape = null;
    this.lastResultTime = 0;
    this.pendingSentAtByModel = {};
    this.pendingFrameByModel = {};
    this.pendingFrameSeqByModel = {};
    this.awaitingByModel = {};
    this.inferSeq = 0;
    this.resultSeq = 0;
    this.latestAnnotatedFrame = null;
    this.latestAnnotatedFrameShape = null;
    this.latestAnnotatedFrameAt = 0;
    this.syncRtspBoxes = cfg.syncRtspBoxes ?? alignedBoxesModeEnabled(this.overlayMode);
    this.serverSideInference = cfg.type === 'server_rtsp';
    this.lastServerMs = null;
    this.lastClientMs = null;
    this.lastTotalMs = null;
    this.lastTritonMs = null;
    this.lastPostMs = null;
    this.bytesThisSec = 0;
    this.totalBytes = 0;
    this.bandwidthBps = 0;
    this.uploadBytesThisSec = 0;
    this.uploadBandwidthBps = 0;
    this.resultBytesThisSec = 0;
    this.resultBandwidthBps = 0;
    this.bandwidthLast = Date.now();

    this.fpsCount = 0;
    this.fpsLast  = Date.now();
    this.fpsDisplay = 0;
    this.previewFpsCount = 0;
    this.previewFpsLast = Date.now();
    this.previewFpsDisplay = 0;
    this.expandedDetectionGroups = new Set();
    this.generation = 0;
    this.inferGeneration = 0;
    this._placeholderPollId = null;
    this._resizeObserver = null;
  }

  _bindCanvas() {
    const el = document.getElementById(`canvas-${this.id}`);
    if (el && document.body.contains(el)) {
      this.canvas = el;
      return true;
    }
    return !!(this.canvas && document.body.contains(this.canvas));
  }

  _teardownTileObservers() {
    if (this._placeholderPollId) {
      clearInterval(this._placeholderPollId);
      this._placeholderPollId = null;
    }
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
  }

  _invalidateAnnotatedFrame() {
    this.latestAnnotatedFrame = null;
    this.latestAnnotatedFrameAt = 0;
  }

  _watchTileLayout() {
    if (typeof ResizeObserver === 'undefined') return;
    const wrap = document.getElementById(`canvas-${this.id}`)?.parentElement;
    if (!wrap) return;
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this._resizeObserver = new ResizeObserver(() => {
      if (!this.active) return;
      this._bindCanvas();
      this._invalidateAnnotatedFrame();
    });
    this._resizeObserver.observe(wrap);
  }

  _unstickInferSockets() {
    this.inferWsList.forEach(ws => {
      if (ws.readyState === 1) ws._awaiting = false;
    });
    Object.keys(this.awaitingByModel).forEach(k => { this.awaitingByModel[k] = false; });
  }

  async start() {
    this.active = true;
    this.generation++;
    this._renderTile();
    await this._startSource();
    updateStreamTotalStats();
  }

  _resetInferenceState() {
    this.pendingSentAtByModel = {};
    this.pendingFrameByModel = {};
    this.pendingFrameSeqByModel = {};
    this.awaitingByModel = {};
    this.annsByModel = {};
    this.lastAnns = [];
    this.lastImgShape = null;
    this.lastResultTime = 0;
    this.inferSeq = 0;
    this.resultSeq = 0;
    this.latestAnnotatedFrame = null;
    this.latestAnnotatedFrameShape = null;
    this.latestAnnotatedFrameAt = 0;
  }

  async _startSource() {
    if (this.type === 'webcam') {
      try {
        const vidConstraint = this.deviceId === '__env__' ? { facingMode: { exact: 'environment' } }
                            : this.deviceId === '__user__' ? { facingMode: { exact: 'user' } }
                            : this.deviceId ? { deviceId: { exact: this.deviceId } }
                            : true;
        try {
          this.mediaStream = await navigator.mediaDevices.getUserMedia({ video: vidConstraint, audio: true });
        } catch {
          this.mediaStream = await navigator.mediaDevices.getUserMedia({ video: vidConstraint, audio: false });
        }
        this.videoEl = document.createElement('video');
        this.videoEl.srcObject = this.mediaStream;
        this.videoEl.autoplay = true; this.videoEl.muted = this.audioMuted;
        this.videoEl.playsInline = true;
        let _wcStarted = false;
        const _wcGo = () => { if (!_wcStarted) { _wcStarted = true; this._connectInfer(); this._startVideoLoop(); } };
        this.videoEl.onloadedmetadata = _wcGo;
        this.videoEl.play().then(_wcGo).catch(() => {});
      } catch(e) {
        toast('Webcam error: ' + e.message, 'error');
        this._setStoppedUI();
      }

    } else if (this.type === 'file') {
      this.videoEl = document.createElement('video');
      this.videoEl.src = URL.createObjectURL(this.src);
      this.videoEl.loop = true; this.videoEl.muted = this.audioMuted;
      this.videoEl.onloadeddata = () => {
        this.videoEl.play();
        this._connectInfer();
        this._startVideoLoop();
      };
      this.videoEl.load();

    } else if (this.type === 'hls') {
      this.videoEl = document.createElement('video');
      this.videoEl.src = this.src;
      this.videoEl.autoplay = true; this.videoEl.muted = this.audioMuted;
      this.videoEl.crossOrigin = 'anonymous';
      this.videoEl.onloadeddata = () => {
        this._connectInfer();
        this._startVideoLoop();
      };
      this.videoEl.onerror = () => toast(`HLS error for "${this.name}"`, 'error');

    } else if (this.type === 'ws') {
      this._connectInfer();
      this._connectSourceWs();
    } else if (this.type === 'server_rtsp') {
      await this._startManagedRtsp();
    }
  }

  async _startManagedRtsp() {
    HOST = document.getElementById('host-input').value.replace(/\/$/, '');
    try {
      const textParams = splitLiveTextParams(this.models, this.classes);
      const resp = await apiFetch('/streams', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          name: this.name,
          url: this.src,
          models: this.models,
          expand_ensembles: true,
          classes: textParams.classes,
          prompts: textParams.prompts,
          imgsz: String(this.imgsz),
          conf: this.conf,
          fps: this.fps,
          preview_fps: this.previewFps,
          max_result_age_ms: 3000,
          source_max_height: Number(this.sourceMaxHeight) || 0,
          backend: this.rtspBackend,
          jpeg_quality: 70,
          live_transport: alignedBoxesModeEnabled(this.overlayMode) ? 'api_jpeg' : (this.preferredLiveTransport || 'go2rtc'),
          enable_tracking: this.enableTracking,
          enable_recording: this.enableRecording,
          // Send the canonical annotated_preview flag so the API creates the stream
          // with the right server-side drawing mode from the very first frame.
          annotated_preview: alignedBoxesModeEnabled(this.overlayMode),
        }),
      });
      this.managedStreamId = resp.id;
      this.liveTransport = resp.live_transport || 'api_jpeg';
      this.go2rtcName = resp.go2rtc_name || null;
      this.go2rtcPublicUrl = resp.go2rtc_public_url || null;
      // Sync annotatedPreview from the API response — this is the ground truth
      // for whether the server is baking boxes into the preview JPEG.
      // Derive from overlayMode (not from resp.annotated_preview) to ensure
      // client and server are always in agreement regardless of saved config.
      this.annotatedPreview = this.type === 'server_rtsp' && alignedBoxesModeEnabled(this.overlayMode);
      if (!alignedBoxesModeEnabled(this.overlayMode) && this.liveTransport === 'go2rtc' && this.go2rtcName && this.go2rtcPublicUrl) {
        this._connectGo2RtcWebRtc();
      } else {
        if (!alignedBoxesModeEnabled(this.overlayMode) && resp.go2rtc_error) {
          console.warn(`[go2rtc] unavailable for "${this.name}", using API preview: ${resp.go2rtc_error}`);
        }
        this._connectManagedPreview();
      }
      this._connectManagedEvents();
      this._setBorder('live');
    } catch(e) {
      toast(`Managed RTSP error for "${this.name}": ${e.message}`, 'error', 7000);
      this._setBorder('error');
      this._setStoppedUI();
    }
  }

  _go2RtcUrl(path) {
    let base = String(this.go2rtcPublicUrl || '').replace(/\/$/, '');
    if (!base) {
      base = `${HOST.replace(/\/$/, '')}/go2rtc`;
    } else if (!base.endsWith('/go2rtc') && !base.includes(':1984')) {
      base = `${base}/go2rtc`;
    }
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${base}${cleanPath}`;
  }

  async _connectGo2RtcWebRtc() {
    if (this.previewWatchdogId) {
      clearInterval(this.previewWatchdogId);
      this.previewWatchdogId = null;
    }
    if (!this.go2rtcName || !this.go2rtcPublicUrl) return;
    const wrap = document.getElementById(`canvas-${this.id}`)?.parentElement;
    if (!wrap) return;
    wrap.classList.add('go2rtc-live');
    let video = document.getElementById(`go2rtc-video-${this.id}`);
    if (!video) {
      video = document.createElement('video');
      video.id = `go2rtc-video-${this.id}`;
      video.className = 'tile-live-video';
      video.autoplay = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.muted = this.audioMuted;
      wrap.insertBefore(video, this.canvas || wrap.firstChild);
    }
    this.go2rtcVideo = video;
    this.videoEl = video;

    // Live-edge stall recovery: if the video has not advanced for 2 s
    // (WebRTC decode stall or browser media-timeline sync issue), nudge
    // currentTime to the buffered end to resume smooth playback.
    if (this._liveEdgeGuardId) clearInterval(this._liveEdgeGuardId);
    let _lastTime = -1;
    let _stuckCount = 0;
    let _coldStartCount = 0;
    this._liveEdgeGuardId = setInterval(() => {
      if (!this.active || !this.go2rtcVideo) return;
      const v = this.go2rtcVideo;
      
      // Cold-start stall recovery: WebRTC connection established but video remains black/not decoding
      if (v.readyState < 2 || !v.videoWidth || !v.videoHeight) {
        _coldStartCount++;
        if (_coldStartCount >= 5) { // 5 seconds of black screen
          console.warn(`[WebRTC] Cold-start stalled (readyState ${v.readyState}, dim ${v.videoWidth}x${v.videoHeight}). Reconnecting...`);
          _coldStartCount = 0;
          this._scheduleManagedPreviewReconnect(0);
        }
        return;
      }
      _coldStartCount = 0;

      if (v.paused) return;
      if (v.currentTime === _lastTime) {
        _stuckCount++;
        if (_stuckCount >= 2 && v.buffered.length > 0) {
          // Seek to live edge
          const liveEdge = v.buffered.end(v.buffered.length - 1);
          if (liveEdge - v.currentTime > 0.5) {
            v.currentTime = liveEdge - 0.1;
          } else {
            v.play().catch(() => {});
          }
          _stuckCount = 0;
        }
      } else {
        _lastTime = v.currentTime;
        _stuckCount = 0;
      }
    }, 1000);

    try {
      const pc = new RTCPeerConnection({
        bundlePolicy: 'max-bundle',
        iceTransportPolicy: 'all',
      });
      this.go2rtcPeer = pc;
      const videoTransceiver = pc.addTransceiver('video', {direction: 'recvonly'});
      pc.addTransceiver('audio', {direction: 'recvonly'});

      // Prefer H.264 baseline/high so go2rtc can pass through the camera's
      // native H.264 stream without re-encoding (zero transcode = native FPS).
      if (videoTransceiver.setCodecPreferences) {
        try {
          const caps = RTCRtpReceiver.getCapabilities?.('video')?.codecs || [];
          const h264 = caps.filter(c => c.mimeType.toLowerCase().includes('h264'));
          const others = caps.filter(c => !c.mimeType.toLowerCase().includes('h264'));
          if (h264.length) videoTransceiver.setCodecPreferences([...h264, ...others]);
        } catch {}
      }

      pc.ontrack = (ev) => {
        const stream = ev.streams?.[0];
        if (!stream) return;
        video.srcObject = stream;
        video.play().catch(() => {});
        // Hide placeholder as soon as we get a track (before metadata)
        const ph = document.getElementById(`ph-${this.id}`);
        if (ph) ph.style.display = 'none';
      };
      video.onloadedmetadata = () => {
        const w = video.videoWidth || 1280;
        const h = video.videoHeight || 720;
        wrap.style.aspectRatio = `${w} / ${h}`;
        if (this.canvas) {
          this.canvas.width = w;
          this.canvas.height = h;
        }
        // Ensure canvas is last child (on top of video in stacking order)
        if (this.canvas && wrap.lastElementChild !== this.canvas) {
          wrap.appendChild(this.canvas);
        }
        this._drawGo2RtcOverlay();
      };
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await new Promise(resolve => {
        if (pc.iceGatheringState === 'complete') return resolve();
        const done = () => {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', done);
            resolve();
          }
        };
        pc.addEventListener('icegatheringstatechange', done);
        setTimeout(resolve, 1200);
      });
      const ac = new AbortController();
      const fetchTimeout = setTimeout(() => ac.abort(), 10000); // 10s timeout
      let resp;
      try {
        resp = await fetch(this._go2RtcUrl(`/api/webrtc?src=${encodeURIComponent(this.go2rtcName)}`), {
          method: 'POST',
          headers: {'Content-Type': 'application/sdp'},
          body: pc.localDescription.sdp,
          signal: ac.signal,
        });
      } finally {
        clearTimeout(fetchTimeout);
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
      await pc.setRemoteDescription({type: 'answer', sdp: await resp.text()});
      this._startGo2RtcFrameCounter();
      const ph2 = document.getElementById(`ph-${this.id}`);
      if (ph2) ph2.style.display = 'none';
      this._setBorder('live');
    } catch (e) {
      console.warn(`[WebRTC] go2rtc WebRTC failed for "${this.name}", falling back to API preview: ${e.message}`);
      this._closeGo2Rtc();
      this._connectManagedPreview();
    }
  }

  _startGo2RtcFrameCounter() {
    const video = this.go2rtcVideo;
    if (!video) return;
    if (typeof video.requestVideoFrameCallback !== 'function') {
      const loop = () => {
        if (!this.active || !this.go2rtcVideo) return;
        this._drawGo2RtcOverlay();
        this.go2rtcOverlayLoopId = requestAnimationFrame(loop);
      };
      this.go2rtcOverlayLoopId = requestAnimationFrame(loop);
      return;
    }
    const tick = () => {
      if (!this.active || !this.go2rtcVideo) return;
      this._drawGo2RtcOverlay();
      this.previewFpsCount++;
      const now = Date.now();
      if (now - this.previewFpsLast >= 1000) {
        this.previewFpsDisplay = this.previewFpsCount;
        this.previewFpsCount = 0;
        this.previewFpsLast = now;
        const fpsEl = document.getElementById(`fps-${this.id}`);
        if (fpsEl) fpsEl.textContent = `${this.previewFpsDisplay} native fps`;
        updateStreamTotalStats();
      }
      this.go2rtcFrameCb = video.requestVideoFrameCallback(tick);
    };
    this.go2rtcFrameCb = video.requestVideoFrameCallback(tick);
  }

  _drawGo2RtcOverlay() {
    if (!this.active || !this.go2rtcVideo || !this._bindCanvas()) return;
    const v = this.go2rtcVideo;
    // Use the video's intrinsic resolution as the canvas coordinate space.
    // CSS handles the actual display scaling via width:100%/height:100%.
    const w = v.videoWidth || this.canvas.width || 1280;
    const h = v.videoHeight || this.canvas.height || 720;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    if (this.lastAnns.length && this.lastImgShape && Date.now() - this.lastResultTime <= 1200) {
      drawAnnotations(ctx, w, h, this.lastAnns, this.lastImgShape, this.catColorMap);
    }
  }

  _closeGo2Rtc() {
    if (this._liveEdgeGuardId) {
      clearInterval(this._liveEdgeGuardId);
      this._liveEdgeGuardId = null;
    }
    if (this.go2rtcOverlayLoopId) {
      cancelAnimationFrame(this.go2rtcOverlayLoopId);
      this.go2rtcOverlayLoopId = null;
    }
    if (this.go2rtcPeer) {
      try { this.go2rtcPeer.close(); } catch {}
      this.go2rtcPeer = null;
    }
    if (this.go2rtcVideo) {
      try {
        const stream = this.go2rtcVideo.srcObject;
        stream?.getTracks?.().forEach(t => t.stop());
        this.go2rtcVideo.pause();
        this.go2rtcVideo.srcObject = null;
        this.go2rtcVideo.remove();
      } catch {}
      this.go2rtcVideo = null;
    }
    const wrap = document.getElementById(`canvas-${this.id}`)?.parentElement;
    if (wrap) {
      wrap.classList.remove('go2rtc-live');
      wrap.style.aspectRatio = '';
    }
  }

  _connectManagedPreview() {
    if (!this.managedStreamId) return;
    const hostVal = document.getElementById('host-input')?.value?.trim();
    const baseHost = (hostVal && /^https?:\/\//i.test(hostVal)) ? hostVal.replace(/\/$/, '') : window.location.origin.replace(/\/$/, '');
    const wsHost = baseHost.replace(/^http/, 'ws');
    const streamId = this.id;
    const gen = this.generation;
    const ws = new WebSocket(`${wsHost}/streams/${this.managedStreamId}/preview`);
    this.previewWs = ws;
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      const inst = streams.get(streamId);
      if (!inst) return;
      inst.lastSrcFrameAt = Date.now();
      inst._setBorder('live');
      inst._startManagedPreviewWatchdog();
    };
    ws.onmessage = (e) => {
      const inst = streams.get(streamId);
      if (!inst?.active || gen !== inst.generation) return;
      inst._onManagedPreview(e.data);
    };
    ws.onerror = () => streams.get(streamId)?._setBorder('error');
    ws.onclose = (e) => {
      const inst = streams.get(streamId);
      if (!inst) return;
      if (inst.active && !ws._expectedClose && e.code !== 1000) {
        console.warn(`[Preview WS] Closed for "${inst.name}": ${e.reason || `code ${e.code}`}`);
        inst._setBorder('error');
        inst._scheduleManagedPreviewReconnect();
      }
    };
  }

  _connectManagedEvents() {
    if (!this.managedStreamId) return;
    const hostVal = document.getElementById('host-input')?.value?.trim();
    const baseHost = (hostVal && /^https?:\/\//i.test(hostVal)) ? hostVal.replace(/\/$/, '') : window.location.origin.replace(/\/$/, '');
    const wsHost = baseHost.replace(/^http/, 'ws');
    const streamId = this.id;
    const gen = this.generation;
    const ws = new WebSocket(`${wsHost}/streams/${this.managedStreamId}/events`);
    this.eventWs = ws;
    ws.onmessage = (e) => {
      const inst = streams.get(streamId);
      if (!inst?.active || gen !== inst.generation) return;
      try {
        inst.resultBytesThisSec += new Blob([e.data]).size;
        inst._updateBandwidthStats();
        const data = JSON.parse(e.data);
        if (data.type === 'detections') inst._onManagedEvent(data);
      } catch {}
    };
    ws.onerror = () => streams.get(streamId)?._setBorder('error');
    ws.onclose = (e) => {
      const inst = streams.get(streamId);
      if (!inst) return;
      if (inst.active && !ws._expectedClose && e.code !== 1000) {
        console.warn(`[Events WS] Closed for "${inst.name}": ${e.reason || `code ${e.code}`}`);
        inst._setBorder('error');
        inst._scheduleManagedPreviewReconnect();
      }
    };
  }

  _startManagedPreviewWatchdog() {
    if (this.type !== 'server_rtsp' || this.previewWatchdogId) return;
    this.previewWatchdogId = setInterval(() => {
      if (!this.active || this.type !== 'server_rtsp' || !this.srcAutoReconnect) return;
      const age = Date.now() - (this.lastSrcFrameAt || 0);
      if (this.lastSrcFrameAt && age > this.srcStallTimeoutMs) {
        console.info(`[NVR Client] "${this.name}" preview stalled ${Math.round(age / 1000)}s, reconnecting`);
        this._scheduleManagedPreviewReconnect(0);
      }
    }, 1000);
  }

  _scheduleManagedPreviewReconnect(delayMs = 1000) {
    if (!this.active || this.type !== 'server_rtsp' || !this.managedStreamId) return;
    if ((this.srcReconnects || 0) >= 2) {
      console.warn(`[NVR Client] Max reconnect attempts reached for "${this.name}". Showing restart button.`);
      this.srcReconnects = 0;
      this._setBorder('error');
      this._setStoppedUI();
      return;
    }
    clearTimeout(this.previewReconnectTimer);
    this.previewReconnectTimer = setTimeout(() => {
      if (!this.active || !this.managedStreamId) return;
      this.srcReconnects = (this.srcReconnects || 0) + 1;
      if (this.previewWs) { this.previewWs._expectedClose = true; this.previewWs.close(); this.previewWs = null; }
      if (this.eventWs) { this.eventWs._expectedClose = true; this.eventWs.close(); this.eventWs = null; }
      this._closeGo2Rtc();
      
      if (this.preferredLiveTransport === 'go2rtc' && !alignedBoxesModeEnabled(this.overlayMode)) {
        this._connectGo2RtcWebRtc();
      } else {
        this._connectManagedPreview();
      }
      this._connectManagedEvents();
    }, delayMs);
  }

  _connectSourceWs(isReconnect = false) {
    try {
      if (this.srcWs) {
        this.srcWs._expectedClose = true;
        this.srcWs.close();
        this.srcWs = null;
      }
      this.srcWs = new WebSocket(this.src);
      this.srcWs.binaryType = 'arraybuffer';
      this.srcWs.onopen = () => {
        this.lastSrcFrameAt = Date.now();
        this._setBorder('live');
        this._startSourceWatchdog();
        if (isReconnect) toast(`"${this.name}" source reconnected`, 'success', 2500);
      };
      this.srcWs.onerror = () => toast(`WS error for "${this.name}"`, 'error');
      this.srcWs.onclose = (e) => {
        const sock = e.currentTarget || this.srcWs;
        if (this.active && !sock?._expectedClose && e.code !== 1000) {
          toast(`WS closed for "${this.name}": ${e.reason || `code ${e.code}`}`, 'error', 7000);
          this._setBorder('error');
          this._scheduleSourceReconnect();
        }
      };
      const streamId = this.id;
      this.srcWs.onmessage = (e) => {
        const inst = streams.get(streamId);
        if (inst?.active) inst._onWsFrame(e.data);
      };
    } catch(e) {
      toast('WS connect error: ' + e.message, 'error');
      this._scheduleSourceReconnect();
    }
  }

  _startSourceWatchdog() {
    if (this.type !== 'ws' || this.srcWatchdogId) return;
    this.srcWatchdogId = setInterval(() => {
      if (!this.active || !this.srcAutoReconnect) return;
      const age = Date.now() - (this.lastSrcFrameAt || 0);
      if (this.lastSrcFrameAt && age > this.srcStallTimeoutMs) {
        toast(`"${this.name}" source stalled ${Math.round(age / 1000)}s, reconnecting`, 'info', 4000);
        this._reconnectSourceWs();
      }
    }, 1000);
  }

  _scheduleSourceReconnect() {
    if (!this.active || !this.srcAutoReconnect || this.type !== 'ws') return;
    clearTimeout(this._srcReconnectTimer);
    this._srcReconnectTimer = setTimeout(() => this._reconnectSourceWs(), this.srcReconnectDelayMs);
  }

  _reconnectSourceWs() {
    if (!this.active || this.type !== 'ws') return;
    this.srcReconnects++;
    this.lastSrcFrameAt = Date.now();
    this._connectSourceWs(true);
  }

  _deleteManagedStreamQuietly() {
    if (!this.managedStreamId) return;
    const sid = this.managedStreamId;
    this.managedStreamId = null;
    
    // Deduplication check: if another active stream on this page is using the same backend stream,
    // do not issue a DELETE request to the server.
    for (const [id, s] of streams.entries()) {
      if (s !== this && s.managedStreamId === sid && s.active) {
        return;
      }
    }

    HOST = document.getElementById('host-input').value.replace(/\/$/, '');
    fetch(HOST + `/streams/${sid}`, {method:'DELETE'}).catch(() => {});
  }

  // Dừng inference nhưng giữ tile
  stop() {
    if (!this.active) return;
    this.active = false;
    this.generation++;
    this._teardownTileObservers();
    this._resetInferenceState();
    if (this.srcWatchdogId) { clearInterval(this.srcWatchdogId); this.srcWatchdogId = null; }
    if (this.previewWatchdogId) { clearInterval(this.previewWatchdogId); this.previewWatchdogId = null; }
    if (this._srcReconnectTimer) { clearTimeout(this._srcReconnectTimer); this._srcReconnectTimer = null; }
    if (this.previewReconnectTimer) { clearTimeout(this.previewReconnectTimer); this.previewReconnectTimer = null; }
    if (this.loopId)      { cancelAnimationFrame(this.loopId); this.loopId = null; }
    if (this.inferWs)     { this.inferWs._expectedClose = true; this.inferWs.close(); this.inferWs = null; }
    this.inferWsList.forEach(w => { w._expectedClose = true; w.close(); });
    this.inferWsList = [];
    if (this.srcWs)       { this.srcWs._expectedClose = true; this.srcWs.close();   this.srcWs   = null; }
    if (this.previewWs)   { this.previewWs._expectedClose = true; this.previewWs.close(); this.previewWs = null; }
    if (this.eventWs)     { this.eventWs._expectedClose = true; this.eventWs.close(); this.eventWs = null; }
    this._closeGo2Rtc();
    this._deleteManagedStreamQuietly();
    if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }
    if (this.videoEl)     { this.videoEl.pause(); }
    this._setStoppedUI();
    updateStreamTotalStats();
  }

  // Khởi động lại stream đã dừng
  async resume() {
    // Tear down all local connections immediately
    if (this.previewWs) { this.previewWs._expectedClose = true; this.previewWs.close(); this.previewWs = null; }
    if (this.eventWs) { this.eventWs._expectedClose = true; this.eventWs.close(); this.eventWs = null; }
    this._closeGo2Rtc();
    if (this.srcWatchdogId) { clearInterval(this.srcWatchdogId); this.srcWatchdogId = null; }
    if (this.previewWatchdogId) { clearInterval(this.previewWatchdogId); this.previewWatchdogId = null; }
    if (this._srcReconnectTimer) { clearTimeout(this._srcReconnectTimer); this._srcReconnectTimer = null; }
    if (this.previewReconnectTimer) { clearTimeout(this.previewReconnectTimer); this.previewReconnectTimer = null; }
    // Show connecting UI immediately (before any network calls)
    this.active = true;
    this.srcReconnects = 0;
    this.generation++;
    this._resetInferenceState();
    if (this.videoEl && this.type !== 'file') {
      if (this.videoEl.src && this.type !== 'file') URL.revokeObjectURL(this.videoEl.src);
      this.videoEl = null;
    }
    this._setRunningUI();
    // Fire-and-forget DELETE of old stream, then immediately start new one
    if (this.managedStreamId) {
      const sid = this.managedStreamId;
      this.managedStreamId = null;
      HOST = document.getElementById('host-input').value.replace(/\/$/, '');
      fetch(`${HOST}/streams/${sid}`, {method:'DELETE'}).catch(() => {});
    }
    await this._startSource();
    updateStreamTotalStats();
    toast(`"${this.name}" — ${uiLabel('restarted')}`, 'success');
  }

  // Xóa hẳn tile khỏi giao diện
  remove() {
    const tab = this.tab;
    this.active = false;
    this.generation++;
    this._teardownTileObservers();
    this._resetInferenceState();
    if (this.srcWatchdogId) { clearInterval(this.srcWatchdogId); this.srcWatchdogId = null; }
    if (this.previewWatchdogId) { clearInterval(this.previewWatchdogId); this.previewWatchdogId = null; }
    if (this._srcReconnectTimer) { clearTimeout(this._srcReconnectTimer); this._srcReconnectTimer = null; }
    if (this.previewReconnectTimer) { clearTimeout(this.previewReconnectTimer); this.previewReconnectTimer = null; }
    if (this.loopId)      { cancelAnimationFrame(this.loopId); this.loopId = null; }
    if (this.inferWs)     { this.inferWs._expectedClose = true; this.inferWs.close(); this.inferWs = null; }
    this.inferWsList.forEach(w => { w._expectedClose = true; w.close(); });
    this.inferWsList = [];
    if (this.srcWs)       { this.srcWs._expectedClose = true; this.srcWs.close();   this.srcWs   = null; }
    if (this.previewWs)   { this.previewWs._expectedClose = true; this.previewWs.close(); this.previewWs = null; }
    if (this.eventWs)     { this.eventWs._expectedClose = true; this.eventWs.close(); this.eventWs = null; }
    this._closeGo2Rtc();
    this._deleteManagedStreamQuietly();
    if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }
    if (this.videoEl)     { this.videoEl.pause(); if (this.videoEl.src) URL.revokeObjectURL(this.videoEl.src); this.videoEl = null; }
    this.canvas = null;
    const tile = document.getElementById(`tile-${this.id}`);
    if (tile) tile.remove();
    streams.delete(this.id);
    updateStreamEmpty(tab);
    updateStopAllBtn(tab);
    updateStreamTotalStats();
    refreshStreamsAfterGridChange();
  }

  _setStoppedUI() {
    this.active = false;
    const overlay = document.getElementById(`tile-stopped-${this.id}`);
    if (overlay) overlay.classList.add('show');
    const btn = document.getElementById(`tile-stopbtn-${this.id}`);
    if (btn) {
      btn.innerHTML = `<svg style="width:10px;height:10px;fill:currentColor;display:inline-block;vertical-align:-1px;margin-right:4px;" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>${uiLabel('Start')}`;
      btn.style.background = '#ffffff18';
      btn.style.border = '1px solid #ffffff33';
    }
    btn?.removeEventListener('click', btn._stopHandler);
    btn._stopHandler = () => this.resume();
    btn?.addEventListener('click', btn._stopHandler);
    this._setBorder('idle');
    const detEl = document.getElementById(`det-${this.id}`);
    if (detEl) detEl.textContent = '';
    const cntEl = document.getElementById(`tile-det-cnt-${this.id}`);
    if (cntEl) cntEl.textContent = '';
    const detsEl = document.getElementById(`tile-dets-${this.id}`);
    if (detsEl) detsEl.innerHTML = `<div class="tile-no-det">${uiLabel('Stream stopped')}</div>`;
  }

  _setRunningUI() {
    const overlay = document.getElementById(`tile-stopped-${this.id}`);
    if (overlay) overlay.classList.remove('show');
    const btn = document.getElementById(`tile-stopbtn-${this.id}`);
    if (btn) {
      btn.innerHTML = `<svg style="width:10px;height:10px;fill:currentColor;display:inline-block;vertical-align:-1px;margin-right:4px;" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>${uiLabel('Stop')}`;
      btn.style.background = '#f8717199';
      btn.style.border = '';
      btn.onclick = () => this.stop();
    }
    this._setBorder('live');
  }

  toggleAudio() {
    if (!this.videoEl) {
      toast(`Audio is not available for "${this.name}"`, 'info');
      return;
    }
    if (this.mediaStream && !this.mediaStream.getAudioTracks().length) {
      toast(`No audio track for "${this.name}"`, 'info');
      return;
    }
    this.audioMuted = !this.audioMuted;
    this.videoEl.muted = this.audioMuted;
    const btn = document.getElementById(`tile-audio-${this.id}`);
    if (btn) btn.textContent = this.audioMuted ? uiLabel('Audio Off') : uiLabel('Audio On');
  }

  async toggleRecording() {
    if (this.type !== 'server_rtsp' || !this.managedStreamId) return;
    const nextState = !this.enableRecording;
    try {
      const r = await fetch(`${HOST}/api/v1/streams/${this.managedStreamId}/recording`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ enabled: nextState })
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
      this.enableRecording = data.recording_enabled;
      this._updateRecordingUI();
      toast(this.enableRecording ? 'Recording started on server' : 'Recording stopped', 'success');
    } catch(e) {
      toast('Failed to toggle recording: ' + e.message, 'error');
    }
  }

  _updateRecordingUI() {
    const btn = document.getElementById(`tile-record-${this.id}`);
    if (btn) {
      btn.style.background = this.enableRecording ? 'rgba(239,68,68,.15)' : 'transparent';
      btn.style.color = this.enableRecording ? 'var(--red)' : 'var(--text2)';
      btn.style.borderColor = this.enableRecording ? 'var(--red)' : 'var(--border)';
      const dot = btn.querySelector('.record-dot');
      if (dot) {
        dot.style.background = this.enableRecording ? 'var(--red)' : 'var(--text3)';
        dot.style.animation = this.enableRecording ? 'pulse 1.5s infinite' : 'none';
      }
    }
    const bar = document.querySelector(`#tile-${this.id} .tile-status-bar`);
    if (bar) {
      let recBadge = bar.querySelector('.rec-badge');
      if (this.enableRecording) {
        if (!recBadge) {
          recBadge = document.createElement('span');
          recBadge.className = 'rec-badge';
          recBadge.style.cssText = 'background:var(--red);color:#fff;border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700;animation:pulse 1.5s infinite;margin-right:4px;';
          recBadge.textContent = '● REC';
          bar.insertBefore(recBadge, bar.firstChild);
        }
      } else {
        if (recBadge) recBadge.remove();
      }
    }
  }

  _connectInfer() {
    const streamId = this.id;
    const gen = this.generation;
    const inferGen = ++this.inferGeneration;
    HOST = document.getElementById('host-input').value.replace(/\/$/, '');
    const wsHost = HOST.replace(/^http/, 'ws');
    this.inferWsList.forEach(w => { w._expectedClose = true; w.close(); });
    this.inferWsList = [];
    this.inferWs = null;
    this.annsByModel = {};
    const models = this.models.length ? this.models : [this.model].filter(Boolean);
    for (const model of models) {
      if (modelRequiresPrompts(model) && !this.classes) {
        toast(`"${model}" requires YOLOE prompts`, 'error');
        this._setBorder('error');
        return;
      }
    }
    try {
      this.inferWsList = models.map(model => {
        let url = `${wsHost}/ws/stream?model=${encodeURIComponent(model)}&fps=${this.fps}&imgsz=${this.imgsz}&conf=${this.conf}`;
        if (this.enableTracking) url += '&track=true';
        url = appendPromptIfNeeded(url, model, this.classes);
        const ws = new WebSocket(url);
        ws._model = model;
        ws._awaiting = false;
        ws._sentAt = 0;
        ws.binaryType = 'arraybuffer';
        ws.onopen  = () => streams.get(streamId)?._setBorder('live');
        ws.onmessage = (e) => {
          const inst = streams.get(streamId);
          if (!inst) return;
          ws._awaiting = false;
          if (!inst.active || gen !== inst.generation || inferGen !== inst.inferGeneration) return;
          try {
            inst.resultBytesThisSec += typeof e.data === 'string'
              ? new Blob([e.data]).size
              : (e.data?.byteLength ?? e.data?.size ?? 0);
            inst._updateBandwidthStats();
            const data = JSON.parse(e.data);
            if (data.dropped) return;
            inst._onInferResult(data, model);
          } catch {}
        };
        ws.onerror = () => {
          const inst = streams.get(streamId);
          if (!inst) return;
          ws._awaiting = false;
          inst.awaitingByModel[model] = false;
          inst._setBorder('error');
        };
        ws.onclose = (e) => {
          const inst = streams.get(streamId);
          if (!inst) return;
          ws._awaiting = false;
          inst.awaitingByModel[model] = false;
          if (inst.active && !ws._expectedClose && e.code !== 1000) {
            toast(`Infer WS closed for "${inst.name}" / ${model}: ${e.reason || `code ${e.code}`}`, 'error', 7000);
            inst._setBorder('error');
          }
        };
        return ws;
      });
      this.inferWs = this.inferWsList[0] || null;
    } catch(e) { toast('Infer WS failed: ' + e.message, 'error'); }
  }

  _sendInferFrame(ab) {
    if (!this.active || !this.inferWsList.length || !streams.has(this.id)) return;
    const sentAt = performance.now();
    const byteLen = ab?.byteLength ?? ab?.size ?? 0;
    let sentCount = 0;
    const frameSeq = ++this.inferSeq;
    this.inferWsList.forEach(ws => {
      const model = ws._model || this.model;
      if (ws.readyState === 1 && !ws._awaiting) {
        ws._awaiting = true;
        ws._sentAt = sentAt;
        this.pendingSentAtByModel[model] = sentAt;
        this.pendingFrameByModel[model] = ab;
        this.pendingFrameSeqByModel[model] = frameSeq;
        this.awaitingByModel[model] = true;
        ws.send(ab);
        sentCount++;
      }
    });
    if (byteLen && sentCount) {
      this.uploadBytesThisSec += byteLen * sentCount;
      this._updateBandwidthStats();
    }
  }

 _startVideoLoop() {
    // sendInterval: how often to send a frame to inference
    let lastSend = 0;
    const cap = document.createElement('canvas');

    const gen = this.generation;
    const loop = (ts) => {
      if (!this.active || gen !== this.generation) return;
      this.loopId = requestAnimationFrame(loop);
      if (!this._bindCanvas()) return;

      const src = this.videoEl;
      if (!src || src.readyState < 2) return;
      const w = src.videoWidth || 640;
      const h = src.videoHeight || 480;
      if (this.canvas.width !== w) this.canvas.width = w;
      if (this.canvas.height !== h) this.canvas.height = h;
      if (cap.width !== w) cap.width = w;
      if (cap.height !== h) cap.height = h;
      const ctx = this.canvas.getContext('2d');
      // Step 1: draw display frame at requestAnimationFrame speed.
      const holdAnnotatedFrame = this.syncRtspBoxes && this.latestAnnotatedFrame &&
        Date.now() - this.latestAnnotatedFrameAt < 2000;
      if (holdAnnotatedFrame) {
        ctx.drawImage(this.latestAnnotatedFrame, 0, 0, w, h);
      } else {
        ctx.drawImage(src, 0, 0, w, h);
      }

      // Step 2: overlay latest annotations so labels stay tight to objects every frame.
      // Webcam/file streams follow Detect-tab behavior: keep the latest model result
      // until a newer response arrives. RTSP source streams still clear quickly.
      if (!this.syncRtspBoxes && this.lastAnns.length && this.lastImgShape) {
        drawAnnotations(ctx, w, h, this.lastAnns, this.lastImgShape, this.catColorMap);
      }

      // Step 3: send clean frame to inference server at fps rate.
      const effectiveInterval = 1000 / Math.max(1, this.fps);
      if (ts - lastSend >= effectiveInterval) {
        lastSend = ts;
        const capCtx = cap.getContext('2d');
        capCtx.drawImage(src, 0, 0, w, h);
        cap.toBlob(blob => {
          if (!blob || !this.active || gen !== this.generation) return;
          blob.arrayBuffer().then(ab => {
            if (this.active && gen === this.generation) this._sendInferFrame(ab);
          });
        }, 'image/jpeg', 0.8);
      }
    };
    this.loopId = requestAnimationFrame(loop);
  }

  async _onWsFrame(data) {
    if (!this.active) return;
    if (!this._bindCanvas()) return;
    const gen = this.generation;
    if (!this._wsFrameN) this._wsFrameN = 0;
    this._wsFrameN++;
    this.lastSrcFrameAt = Date.now();
    const byteLen = data?.byteLength ?? data?.size ?? 0;
    if (byteLen) {
      this.bytesThisSec += byteLen;
      this.totalBytes += byteLen;
    }
    try {
      const bitmap = await createImageBitmap(new Blob([data], {type:'image/jpeg'}));
      if (!this.active || gen !== this.generation) {
        bitmap.close();
        return;
      }
      if (this.canvas.width !== bitmap.width) this.canvas.width = bitmap.width;
      if (this.canvas.height !== bitmap.height) this.canvas.height = bitmap.height;
      const ctx = this.canvas.getContext('2d');
      const holdAnnotatedFrame = this.syncRtspBoxes && this.latestAnnotatedFrame &&
        Date.now() - this.latestAnnotatedFrameAt < 2000;
      if (!holdAnnotatedFrame) {
        ctx.drawImage(bitmap, 0, 0);
      } else {
        ctx.drawImage(this.latestAnnotatedFrame, 0, 0, this.canvas.width, this.canvas.height);
      }
      bitmap.close();
      // Forward clean frame to inference server at fps rate, before drawing annotations.
      const now = performance.now();
      const minInterval = 1000 / Math.max(1, this.fps);
      if (!this._lastWsInferSend) this._lastWsInferSend = 0;
      if (this.inferWsList.some(ws => ws.readyState === 1) && now - this._lastWsInferSend >= minInterval) {
        this._lastWsInferSend = now;
        this._sendInferFrame(data);
      }
      if (this.lastAnns.length && this.lastImgShape) {
        const stale = Date.now() - this.lastResultTime > 3000;
        if (stale) {
          this.lastAnns = [];
          this.annsByModel = {};
          this._invalidateAnnotatedFrame();
        } else if (!holdAnnotatedFrame || !this.syncRtspBoxes) {
          drawAnnotations(ctx, this.canvas.width, this.canvas.height, this.lastAnns, this.lastImgShape, this.catColorMap);
        }
      }
      this._updateBandwidthStats();
    } catch {}
  }

  async _onManagedPreview(data) {
    if (!this.active) return;
    if (!this._bindCanvas()) return;
    const gen = this.generation;
    this.lastSrcFrameAt = Date.now();
    let imageData = data;
    if (typeof data === 'string') {
      try {
        const msg = JSON.parse(data);
        if (msg.type === 'preview' && msg.image_b64) {
          imageData = base64ToArrayBuffer(msg.image_b64);
        }
      } catch {}
    }
    const byteLen = imageData?.byteLength ?? imageData?.size ?? 0;
    if (byteLen) {
      this.bytesThisSec += byteLen;
      this.totalBytes += byteLen;
    }
    try {
      await drawJpegBytesToCanvas(this.canvas, imageData);
      if (!this.active || gen !== this.generation) return;

      // annotatedPreview=true: server already drew boxes into this JPEG (source_max_height
      // or exact-boxes mode). Display as-is and stop — never draw client-side annotations
      // on top, regardless of what lastAnns contains.
      if (this.annotatedPreview) {
        const ph = document.getElementById(`ph-${this.id}`);
        if (ph) ph.style.display = 'none';
        this._setBorder('live');
        this.previewFpsCount++;
        const now = Date.now();
        if (now - this.previewFpsLast >= 1000) {
          this.previewFpsDisplay = this.previewFpsCount;
          this.previewFpsCount = 0;
          this.previewFpsLast = now;
          const fpsEl = document.getElementById(`fps-${this.id}`);
          if (fpsEl) fpsEl.textContent = `${this.previewFpsDisplay} preview fps`;
        }
        this._updateBandwidthStats();
        return;
      }

      // Native FPS Live (annotatedPreview=false): draw client-side annotations from JSON.
      const staleOverlay = this.lastResultTime && Date.now() - this.lastResultTime > 1000;
      if (staleOverlay) {
        this.lastAnns = [];
        this.annsByModel = {};
      }
      if (!staleOverlay && this.lastAnns.length && this.lastImgShape) {
        const ctx = this.canvas.getContext('2d');
        if (ctx) {
          drawAnnotations(ctx, this.canvas.width, this.canvas.height, this.lastAnns, this.lastImgShape, this.catColorMap);
        }
      }
      const ph = document.getElementById(`ph-${this.id}`);
      if (ph) ph.style.display = 'none';
      this._setBorder('live');
      this.previewFpsCount++;
      const now = Date.now();
      if (now - this.previewFpsLast >= 1000) {
        this.previewFpsDisplay = this.previewFpsCount;
        this.previewFpsCount = 0;
        this.previewFpsLast = now;
        const fpsEl = document.getElementById(`fps-${this.id}`);
        if (fpsEl) fpsEl.textContent = `${this.previewFpsDisplay} preview fps`;
      }
      this._updateBandwidthStats();
    } catch (e) {
      toast(`Preview decode error for "${this.name}": ${e.message}`, 'error', 5000);
    }
  }

  _onManagedEvent(data) {
    if (!this.active) return;
    if (data?.error) {
      toast(`Stream error for "${this.name}": ${data.error}`, 'error', 7000);
      return;
    }
    this.lastServerMs = data.timing_ms?.total ?? null;
    this.lastTotalMs = this.lastServerMs;
    this.lastClientMs = null;
    this.lastTritonMs = null;
    this.lastPostMs = null;
    this.lastAnns = normalizeAnnotations(data.annotations || [], data.models?.[0]);
    this.lastImgShape = data.image_shape;
    this.lastResultTime = Date.now();
    this.lastAnns.forEach(a => {
      const key = annColorKey(a);
      if (!this.catColorMap[key])
        this.catColorMap[key] = COLORS[Object.keys(this.catColorMap).length % COLORS.length];
    });

    const latencyEl = document.getElementById(`lat-${this.id}`);
    if (latencyEl) latencyEl.textContent = _fmtMs(this.lastServerMs);
    const perfEl = document.getElementById(`tile-perf-${this.id}`);
    if (perfEl) perfEl.innerHTML = this._perfHtml(data, data.models?.join(', ') || this.model);
    if (this.go2rtcVideo && !this.annotatedPreview) {
      // annotatedPreview=true means the server bakes boxes into the JPEG preview;
      // drawing again from JSON would double-render all boxes on the canvas.
      this._drawGo2RtcOverlay();
    }

    const detEl = document.getElementById(`det-${this.id}`);
    if (detEl) detEl.textContent = this.lastAnns.length ? `${this.lastAnns.length} det` : '';
    renderTileDetectionList(this);
    this.fpsCount++;
    const now = Date.now();
    if (now - this.fpsLast >= 1000) {
      this.fpsDisplay = this.fpsCount;
      this.fpsCount = 0; this.fpsLast = now;
      const fpsEl = document.getElementById(`fps-${this.id}`);
      if (fpsEl && this.type !== 'server_rtsp') fpsEl.textContent = this.fpsDisplay + ' fps';
    }
    checkStreamFireAlert(this, this.lastAnns);
    checkStreamCrowdAlert(this, this.lastAnns);
    updateStreamTotalStats();
  }

  _updateBandwidthStats(force = false) {
    const now = Date.now();
    const elapsed = now - this.bandwidthLast;
    if (!force && elapsed < 1000) return;
    this.bandwidthBps = elapsed > 0 ? this.bytesThisSec * 1000 / elapsed : 0;
    this.uploadBandwidthBps = elapsed > 0 ? this.uploadBytesThisSec * 1000 / elapsed : 0;
    this.resultBandwidthBps = elapsed > 0 ? this.resultBytesThisSec * 1000 / elapsed : 0;
    this.bytesThisSec = 0;
    this.uploadBytesThisSec = 0;
    this.resultBytesThisSec = 0;
    this.bandwidthLast = now;
    const bwEl = document.getElementById(`tile-bw-${this.id}`);
    const outEl = document.getElementById(`tile-bw-out-${this.id}`);
    const resEl = document.getElementById(`tile-bw-result-${this.id}`);
    if (bwEl) bwEl.textContent = _fmtBandwidth(this.bandwidthBps);
    if (outEl) outEl.textContent = _fmtBandwidth(this.uploadBandwidthBps);
    if (resEl) resEl.textContent = _fmtBandwidth(this.resultBandwidthBps);
    updateStreamTotalStats();
  }

  _onInferResult(data, model = this.model) {
    if (!this.active) return;
    if (data?.error) {
      this.awaitingByModel[model] = false;
      toast(`Inference error for "${this.name}" / ${model}: ${data.error}`, 'error', 7000);
      return;
    }
    const recvAt = performance.now();
    const sentAt = this.pendingSentAtByModel[model];
    const clientMs = sentAt ? recvAt - sentAt : null;
    const serverMs = data.timing_ms?.total ?? data.timing_ms?.inference ?? null;
    this.lastClientMs = clientMs;
    this.lastServerMs = serverMs;
    this.lastTotalMs = clientMs;
    this.lastTritonMs = data.timing_ms?.triton ?? null;
    this.lastPostMs = data.timing_ms?.postprocess ?? null;
    this.awaitingByModel[model] = false;
    const latencyEl = document.getElementById(`lat-${this.id}`);
    if (latencyEl) {
      latencyEl.textContent = _fmtMs(clientMs ?? serverMs);
    }
    const perfEl = document.getElementById(`tile-perf-${this.id}`);
    if (perfEl) perfEl.innerHTML = this._perfHtml(data, model);
    const seq = this.pendingFrameSeqByModel[model] || 0;
    if (seq && seq < this.resultSeq) {
      return;
    }
    if (seq && seq > this.resultSeq) {
      this.resultSeq = seq;
      this.annsByModel = {};
    }
    const anns = normalizeAnnotations(data.annotations, model);
    this.annsByModel[model] = { anns, imgShape: data.image_shape, t: Date.now() };
    Object.entries(this.annsByModel).forEach(([k, v]) => {
      if (Date.now() - v.t > 2000) delete this.annsByModel[k];
    });
    this.lastAnns      = Object.values(this.annsByModel).flatMap(v => v.anns);
    this.lastImgShape  = data.image_shape;
    this.lastResultTime = Date.now();
    this.lastAnns.forEach(a => {
      const key = annColorKey(a);
      if (!this.catColorMap[key])
        this.catColorMap[key] = COLORS[Object.keys(this.catColorMap).length % COLORS.length];
    });
    if (this.syncRtspBoxes && this.type !== 'server_rtsp') {
      this._drawSyncedRtspResult(model);
    }

    // Badge nhỏ trên canvas
    const detEl = document.getElementById(`det-${this.id}`);
    if (detEl) detEl.textContent = this.lastAnns.length ? `${this.lastAnns.length} det` : '';

    renderTileDetectionList(this);

    // FPS counter
    this.fpsCount++;
    const now = Date.now();
    if (now - this.fpsLast >= 1000) {
      this.fpsDisplay = this.fpsCount;
      this.fpsCount = 0; this.fpsLast = now;
      const fpsEl = document.getElementById(`fps-${this.id}`);
      if (fpsEl) fpsEl.textContent = this.fpsDisplay + ' fps';
      updateStreamTotalStats();
    }

    // Fire check mỗi frame
    checkStreamFireAlert(this, this.lastAnns);
    // Crowd check mỗi frame
    checkStreamCrowdAlert(this, this.lastAnns);
    updateStreamTotalStats();
  }

  async _drawSyncedRtspResult(model) {
    const ab = this.pendingFrameByModel[model];
    if (!ab || !this._bindCanvas() || !this.lastImgShape) return;
    const gen = this.generation;
    try {
      const bitmap = await createImageBitmap(new Blob([ab], {type:'image/jpeg'}));
      if (!this.active || gen !== this.generation) {
        bitmap.close();
        return;
      }
      if (this.canvas.width !== bitmap.width) this.canvas.width = bitmap.width;
      if (this.canvas.height !== bitmap.height) this.canvas.height = bitmap.height;
      const ctx = this.canvas.getContext('2d');
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      drawAnnotations(ctx, this.canvas.width, this.canvas.height, this.lastAnns, this.lastImgShape, this.catColorMap);

      const frozen = document.createElement('canvas');
      frozen.width = this.canvas.width;
      frozen.height = this.canvas.height;
      frozen.getContext('2d').drawImage(this.canvas, 0, 0);
      this.latestAnnotatedFrame = frozen;
      this.latestAnnotatedFrameAt = Date.now();
    } catch {}
  }

  _perfHtml(data, model) {
    const timing = data?.timing_ms || {};
    return `
      <div class="infer-stat live">mode: <span>live</span></div>
      <div class="infer-stat">${hintLabel('model', 'Model that produced the latest JSON result.')}: <span>${escHtml(model)}</span></div>
      <div class="infer-stat">${hintLabel('total', 'Browser round trip: upload frame to API, server inference, and JSON response back.')}: <span>${_fmtMs(this.lastTotalMs)}</span></div>
      <div class="infer-stat">${hintLabel('server', 'Server-only processing time reported by the API.')}: <span>${_fmtMs(this.lastServerMs)}</span></div>
      ${this.lastTritonMs != null ? `<div class="infer-stat">${hintLabel('triton', 'Time spent inside NVIDIA Triton inference.')}: <span>${_fmtMs(this.lastTritonMs)}</span></div>` : ''}
      ${this.lastPostMs != null ? `<div class="infer-stat">${hintLabel('post', 'API postprocess time: NMS, masks, labels, and coordinate mapping.')}: <span>${_fmtMs(this.lastPostMs)}</span></div>` : ''}
      ${perModelTimingStatsHtml(data?.per_model_timing_ms)}
      <div class="infer-stat">${hintLabel('preview in', 'Download bandwidth for preview frames. RTSP Camera mode receives these from /streams/{id}/preview; legacy RTSP bridge receives them from /ws/rtsp.')}: <span id="tile-bw-${this.id}">${_fmtBandwidth(this.bandwidthBps)}</span></div>
      <div class="infer-stat">${hintLabel('infer out', 'Upload bandwidth from this browser to API /ws/stream for inference frames. RTSP Camera server-side mode does not upload frames, so this should stay near zero.')}: <span id="tile-bw-out-${this.id}">${_fmtBandwidth(this.uploadBandwidthBps)}</span></div>
      <div class="infer-stat">${hintLabel('json in', 'Download bandwidth for detection JSON results from API to this browser.')}: <span id="tile-bw-result-${this.id}">${_fmtBandwidth(this.resultBandwidthBps)}</span></div>
      <div class="infer-stat">${hintLabel('reconn', 'How many times the RTSP preview/source WebSocket reconnected.')}: <span>${this.srcReconnects}</span></div>
      <div class="infer-stat">${hintLabel('detect fps', 'Detection JSON results received per second.')}: <span>${this.fpsDisplay}</span></div>
      <div class="infer-stat">${hintLabel('infer cap', 'Maximum inference frames per second requested by the client, clamped by API max_fps.')}: <span>${this.fps}</span></div>
      <div class="infer-stat">${hintLabel('overlay', 'Exact-frame mode draws boxes/masks on the same frame used for inference. Native FPS Live shows the newest preview frame and draws latest JSON in the browser.')}: <span>${overlayModeLabel(this.overlayMode)}</span></div>
      ${this.type === 'server_rtsp' ? `<div class="infer-stat">${hintLabel('live path', 'RTSP Native FPS Live uses go2rtc WebRTC when available. Exact Boxes uses API JPEG preview.')}: <span>${escHtml(this.liveTransport || 'api_jpeg')}</span></div>` : ''}
      ${this.type === 'server_rtsp' ? `<div class="infer-stat">${hintLabel('preview fps', 'Measured preview JPEG frames per second received by the browser.')}: <span>${this.previewFpsDisplay}</span></div><div class="infer-stat">${hintLabel('preview cap', 'Configured preview JPEG FPS sent from API to browser.')}: <span>${this.previewFps}</span></div>` : ''}
      <div class="infer-stat">models: <span>${this.models.length}</span></div>
      <div class="infer-stat">${hintLabel('imgsz', 'Inference input size after letterbox resize.')}: <span>${Array.isArray(data?.inference_imgsz) ? data.inference_imgsz.join('×') : this.imgsz}</span></div>
      <div class="infer-stat">${hintLabel('shape', 'Original frame height and width used to map boxes back.')}: <span>${Array.isArray(data?.image_shape) ? data.image_shape.join('×') : '—'}</span></div>`;
  }

  _renderTile() {
    const gridId  = this.tab === 'rtsp' ? 'rtsp-grid'  : 'stream-grid';
    const emptyId = this.tab === 'rtsp' ? 'rtsp-empty' : 'stream-empty';
    const grid    = document.getElementById(gridId);
    const emptyEl = document.getElementById(emptyId);
    if (emptyEl) emptyEl.remove();

    const tile = document.createElement('div');
    tile.className = 'stream-tile sidebar-collapsed';
    tile.id = `tile-${this.id}`;
    tile.innerHTML = `
      <div class="tile-main">
        <div class="tile-canvas-wrap">
          <canvas class="tile-canvas" id="canvas-${this.id}" style="min-height:140px;"></canvas>
          <div class="tile-placeholder" id="ph-${this.id}">
            <span class="spin" style="font-size:18px;">⟳</span><span>Connecting…</span>
          </div>
          <div class="tile-status-bar">
            ${this.enableRecording ? `<span class="rec-badge" style="background:var(--red);color:#fff;border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700;animation:pulse 1.5s infinite;margin-right:4px;">● REC</span>` : ''}
            <span class="live-badge">LIVE</span>
            <span class="fps-badge" id="fps-${this.id}">— fps</span>
            <span class="latency-badge" id="lat-${this.id}">— ms</span>
          </div>
          <span class="tile-det-badge" id="det-${this.id}"></span>
        <div class="tile-stopped-overlay" id="tile-stopped-${this.id}">
          <svg style="width:28px;height:28px;opacity:.5;fill:currentColor;" viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>
          <span>${uiLabel('Stream stopped')}</span>
          <button class="tile-restart-btn" onclick="streams.get('${this.id}')?.resume()" style="display:inline-flex;align-items:center;gap:6px;">
            <svg style="width:12px;height:12px;fill:currentColor;" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            ${uiLabel('Start')}
          </button>
        </div>
        <button class="tile-sb-toggle" id="tile-sb-toggle-${this.id}" title="${uiTitle('Open stream settings')}" onclick="toggleTileSidebar('${this.id}')">
          <svg style="width:12px;height:12px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
        </button>
        <div style="position:absolute;bottom:8px;right:8px;display:flex;align-items:center;gap:6px;z-index:20;">
          ${this.type === 'server_rtsp' ? `
          <button class="tile-action-btn" id="tile-record-${this.id}" onclick="streams.get('${this.id}')?.toggleRecording()" style="position:static;background:${this.enableRecording ? 'rgba(239,68,68,.2)' : 'rgba(0,0,0,0.6)'};color:${this.enableRecording ? '#ef4444' : 'var(--text2)'};border:1px solid ${this.enableRecording ? '#ef4444' : 'var(--border)'};border-radius:4px;padding:3px 8px;font-size:11px;display:inline-flex;align-items:center;">
            <span class="record-dot" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${this.enableRecording ? '#ef4444' : 'var(--text3)'};margin-right:4px;animation:${this.enableRecording ? 'pulse 1.5s infinite' : 'none'};"></span>
            <span>REC</span>
          </button>
          ` : ''}
          <button class="tile-action-btn" id="tile-audio-${this.id}" data-dyn-label="Audio Off" onclick="streams.get('${this.id}')?.toggleAudio()" style="position:static;background:rgba(0,0,0,0.6);border:1px solid var(--border);border-radius:4px;padding:3px 8px;font-size:11px;">${uiLabel('Audio Off')}</button>
          <button class="tile-action-btn" onclick="tileFullscreen('${this.id}')" style="position:static;background:rgba(0,0,0,0.6);border:1px solid var(--border);border-radius:4px;padding:3px 8px;font-size:11px;display:inline-flex;align-items:center;justify-content:center;" title="Fullscreen">
            <svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/></svg>
          </button>
          <button class="tile-stop-btn" id="tile-stopbtn-${this.id}" onclick="streams.get('${this.id}')?.stop()" style="position:static;padding:3px 10px;font-size:11px;border-radius:4px;display:inline-flex;align-items:center;gap:4px;">
            <svg class="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
            <span>${uiLabel('Stop')}</span>
          </button>
        </div>
        </div>
        <div class="tile-footer">
          <span class="tile-name">${escHtml(this.name)}</span>
          <span class="tile-model-tag" id="tile-modeltag-${this.id}">${escHtml(this.models.join(', '))}</span>
          <button class="tile-edit-btn" data-dyn-label="Edit" onclick="toggleTileSidebar('${this.id}', true)">${uiLabel('Edit')}</button>
          <button onclick="confirmRemoveStream('${this.id}')"
            style="margin-left:auto;padding:2px 8px;border:1px solid #f8717144;background:transparent;
            color:var(--red);border-radius:var(--radius);font-family:var(--font-mono);font-size:9px;cursor:pointer;display:inline-flex;align-items:center;gap:3px;"
            title="Delete stream" data-dyn-label="Delete">
            <svg style="width:9px;height:9px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
            ${uiLabel('Delete')}</button>
        </div>
      </div>
      <div class="tile-sidebar collapsed">
        <div class="tile-sb-section">
          <div class="tile-sb-title">
            ${hintLabel('Performance', 'Latency and bandwidth counters. Hidden by default so the config panel stays stable.')}
            <span class="tile-sb-actions">
              <button class="tile-section-toggle" id="tile-perf-toggle-${this.id}" data-dyn-label="Show" onclick="togglePerfSection('${this.id}')">${uiLabel('Show')}</button>
              <button class="tile-edit-btn" data-dyn-label="Close" style="padding:1px 6px;" onclick="toggleTileSidebar('${this.id}', false, true)">${uiLabel('Close')}</button>
            </span>
          </div>
          <div class="tile-perf-body collapsed" id="tile-perf-${this.id}">
            <div class="infer-stat">${hintLabel('time', 'Browser round-trip latency for the latest inference result.')}: <span>—</span></div>
            <div class="infer-stat">${hintLabel('fps', 'Detection results received per second.')}: <span>${this.fps}</span></div>
            <div class="infer-stat">${hintLabel('preview in', 'Download from API to browser. In RTSP Camera mode this is the optional server-side preview stream.')}: <span id="tile-bw-${this.id}">—</span></div>
            <div class="infer-stat">${hintLabel('infer out', 'Upload from browser to API inference WebSocket. RTSP Camera server-side mode should be zero because API reads and infers directly.')}: <span id="tile-bw-out-${this.id}">—</span></div>
            <div class="infer-stat">${hintLabel('json in', 'Detection JSON download from API to browser.')}: <span id="tile-bw-result-${this.id}">—</span></div>
          </div>
        </div>
        <div class="tile-sb-section tile-param-section" id="tile-param-section-${this.id}">
          <div class="tile-sb-title">
            ${hintLabel('Parameters', 'Changing these values reconnects inference with the new settings.')}
            <span class="tile-sb-actions">
              <button class="tile-section-toggle" id="tile-param-toggle-${this.id}" data-dyn-label="Hide" onclick="toggleTileSection('${this.id}', 'param')">${uiLabel('Hide')}</button>
            </span>
          </div>
          <div class="tile-p-row" style="align-items:flex-start;margin-bottom:6px;">
            <span class="tile-p-lbl" data-dyn-label="models" style="padding-top:3px;" title="${uiTitle('Models run against this stream. More models means more WebSockets and more GPU/API load.')}">${uiLabel('models')}</span>
            <div class="model-checklist" style="flex:1;max-height:95px;padding:4px;">
              ${[...allModels,...allEnsembles].map(m =>
                `<label class="model-check-item" style="padding:3px 4px;">
                  <input type="checkbox" name="tm-${this.id}-cb" value="${escHtml(m.name)}"${this.models.includes(m.name)?' checked':''} />
                  <span style="font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(m.name)}</span>
                </label>`
              ).join('')}
            </div>
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="conf" title="${uiTitle('Minimum confidence score. Higher removes weak detections; lower shows more boxes.')}">${uiLabel('conf')}</span>
            <input class="tile-p-inp" id="tc-${this.id}" type="number" min="0" max="1" step="0.05" value="${this.conf}" />
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="imgsz" title="${uiTitle('Input size sent to the model. Larger can improve small objects but costs more latency.')}">${uiLabel('imgsz')}</span>
            <input class="tile-p-inp" id="ti-${this.id}" value="${this.imgsz}" />
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="infer fps" title="${uiTitle('Inference FPS cap. For RTSP Camera mode, API samples this many frames per second from the camera for detection.')}">${uiLabel('infer fps')}</span>
            <input class="tile-p-inp" id="tf-${this.id}" type="number" min="1" ${apiMaxFps > 0 ? `max="${apiMaxFps}"` : 'max="240"'} step="1" value="${clampFpsValue(this.fps, 10)}" onchange="clampFpsInput(this, ${this.fps}, true)" title="${uiTitle('Custom FPS cap. Clamped to API max_fps.')}" />
          </div>
          ${this.type === 'server_rtsp' ? `
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="preview" title="${uiTitle('Preview FPS sent from API back to this browser. Lower saves network bandwidth without reducing detection FPS.')}">${uiLabel('preview')}</span>
            <input class="tile-p-inp" id="tpf-${this.id}" type="number" min="1" ${apiMaxFps > 0 ? `max="${apiMaxFps}"` : 'max="240"'} step="1" value="${clampFpsValue(this.previewFps, 10)}" onchange="clampFpsInput(this, ${this.previewFps}, true)" title="${uiTitle('RTSP preview FPS. Lower saves browser download bandwidth.')}" />
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="source size" title="${uiTitle('Working resolution for this RTSP stream. Lower values reduce decode/JPEG/inference cost and can improve FPS.')}">${uiLabel('source size')}</span>
            <select class="tile-p-inp" id="tsh-${this.id}" title="${uiTitle('Native keeps camera resolution. 720p is a good production default for web analytics.')}">
              ${sourceSizeOptionsHtml(this.sourceMaxHeight)}
            </select>
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="backend" title="${uiTitle('RTSP reader backend. Auto uses the API default.')}">${uiLabel('backend')}</span>
            <select class="tile-p-inp rtsp-backend-select" id="tbk-${this.id}" title="${uiTitle('Use Auto unless testing a specific RTSP backend.')}">
              ${rtspBackendOptionsHtml(this.rtspBackend)}
            </select>
          </div>` : ''}
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="classes" title="${uiTitle('YOLOE text prompts only. Normal YOLO models use labels.json.')}">${uiLabel('classes')}</span>
            <input class="tile-p-inp" id="tcl-${this.id}" value="${escHtml(this.classes||'')}" placeholder="YOLOE only" />
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="overlay" title="${uiTitle('Exact Boxes uses server-annotated inference frames. Native FPS Live shows the newest source frame and overlays latest JSON.')}">${uiLabel('overlay')}</span>
            <select class="tile-p-inp" id="tom-${this.id}" title="${uiTitle('Exact Boxes is aligned to the inference frame but can be lower FPS. Native FPS Live is smoother but boxes can lag by inference latency.')}">
              <option value="exact" ${this.overlayMode === 'exact' ? 'selected' : ''}>${uiLabel('Exact Boxes')}</option>
              <option value="native_exact" ${this.overlayMode === 'native_exact' ? 'selected' : ''}>${uiLabel('Native FPS Live')}</option>
            </select>
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="watch" title="${uiTitle('Reconnect the source WebSocket if frames stop arriving.')}">${uiLabel('watch')}</span>
            <label style="font-family:var(--font-mono);font-size:9px;color:var(--text2);display:flex;align-items:center;gap:5px;">
              <input id="tar-${this.id}" type="checkbox" ${this.srcAutoReconnect ? 'checked' : ''} />
              <span data-dyn-label="auto reconnect">${uiLabel('auto reconnect')}</span>
            </label>
          </div>
          <div class="tile-p-row">
            <span class="tile-p-lbl" data-dyn-label="stall" title="${uiTitle('Seconds without source frames before auto reconnect triggers.')}">${uiLabel('stall')}</span>
            <input class="tile-p-inp" id="tstall-${this.id}" type="number" min="2" max="60" step="1" value="${Math.round(this.srcStallTimeoutMs / 1000)}" title="${uiTitle('Reconnect source if no RTSP/WS frame arrives for this many seconds')}" />
          </div>
          <button class="tile-apply-btn" data-dyn-label="Reconnect Source" onclick="reconnectStreamSource('${this.id}')">${uiLabel('Reconnect Source')}</button>
          <button class="tile-apply-btn" data-dyn-label="Apply" onclick="applyStreamParams('${this.id}')">⟳ ${uiLabel('Apply')}</button>
        </div>
       <div class="tile-sb-section" id="tile-det-section-${this.id}" style="flex:1;display:flex;flex-direction:column;overflow:hidden;border-bottom:none;padding-bottom:0;">
          <div class="tile-sb-title">
            <span data-dyn-label="Detections">${uiLabel('Detections')}</span>
            <span class="tile-sb-count" id="tile-det-cnt-${this.id}"></span>
            <span class="tile-sb-actions">
              <button class="tile-section-toggle" id="tile-det-toggle-${this.id}" data-dyn-label="Hide" onclick="toggleTileSection('${this.id}', 'det')">${uiLabel('Hide')}</button>
            </span>
          </div>
          <div class="tile-dets-list" id="tile-dets-${this.id}">
            <div class="tile-no-det">${uiLabel('Waiting')}…</div>
          </div>
        </div>
      </div>`;
    grid.appendChild(tile);
    this._bindCanvas();
    this._watchTileLayout();

    if (this._placeholderPollId) clearInterval(this._placeholderPollId);
    this._placeholderPollId = setInterval(() => {
      if (!streams.has(this.id)) {
        clearInterval(this._placeholderPollId);
        this._placeholderPollId = null;
        return;
      }
      if (this._bindCanvas() && this.canvas.width > 0) {
        const ph = document.getElementById(`ph-${this.id}`);
        if (ph) ph.style.display = 'none';
        this._setBorder('live');
        clearInterval(this._placeholderPollId);
        this._placeholderPollId = null;
      }
    }, 200);
  }

  _setBorder(state) {
    const tile = document.getElementById(`tile-${this.id}`);
    if (!tile) return;
    if (state === 'live')  tile.style.borderColor = '#444444';
    else if (state === 'error') tile.style.borderColor = 'var(--red)';
    else tile.style.borderColor = 'var(--border)';
  }
}

/* ══════════════════════════════════════════════════════════════
   STREAM UI HELPERS
══════════════════════════════════════════════════════════════ */
function setGridCols(tab, cols) {
  const gridId = tab === 'rtsp' ? 'rtsp-grid' : 'stream-grid';
  const btnGroupId = tab + '-layout-btns';
  document.getElementById(gridId).style.gridTemplateColumns = `repeat(${cols},1fr)`;
  document.querySelectorAll(`#${btnGroupId} .layout-btn`).forEach((b,i) => {
    b.classList.toggle('active', i+1 === cols);
  });
}

function toggleTileSidebar(id, forceOpen = false, forceClose = false) {
  const tile = document.getElementById(`tile-${id}`);
  const sidebar = document.querySelector(`#tile-${id} .tile-sidebar`);
  const btn     = document.getElementById(`tile-sb-toggle-${id}`);
  if (!sidebar) return;
  const collapsed = forceClose ? true : (forceOpen ? false : sidebar.classList.toggle('collapsed'));
  if (forceOpen) sidebar.classList.remove('collapsed');
  if (forceClose) sidebar.classList.add('collapsed');
  tile?.classList.toggle('sidebar-collapsed', sidebar.classList.contains('collapsed'));
  tile?.classList.toggle('sidebar-open', !sidebar.classList.contains('collapsed'));
  if (btn) btn.innerHTML = collapsed 
    ? '<svg style="width:12px;height:12px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>'
    : '<svg style="width:12px;height:12px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  btn?.setAttribute('title', collapsed ? uiTitle('Open stream settings') : uiTitle('Close stream settings'));
  requestAnimationFrame(() => {
    const inst = streams.get(id);
    if (inst?.canvas && inst.lastAnns.length && inst.lastImgShape) {
      const ctx = inst.canvas.getContext('2d');
      drawAnnotations(ctx, inst.canvas.width, inst.canvas.height, inst.lastAnns, inst.lastImgShape, inst.catColorMap);
    }
  });
}

function togglePerfSection(id) {
  const el = document.getElementById(`tile-perf-${id}`);
  const btn = document.getElementById(`tile-perf-toggle-${id}`);
  if (!el) return;
  const collapsed = el.classList.toggle('collapsed');
  if (btn) btn.textContent = collapsed ? uiLabel('Show') : uiLabel('Hide');
}

function toggleTileSection(id, section) {
  const el = document.getElementById(`tile-${section}-section-${id}`);
  const btn = document.getElementById(`tile-${section}-toggle-${id}`);
  const sidebar = document.querySelector(`#tile-${id} .tile-sidebar`);
  if (!el) return;
  const collapsed = el.classList.toggle('section-collapsed');
  if (sidebar) {
    sidebar.classList.toggle(`${section === 'det' ? 'detections' : 'parameters'}-collapsed`, collapsed);
  }
  if (btn) btn.textContent = collapsed ? uiLabel('Show') : uiLabel('Hide');
}

async function reconnectStreamSource(id) {
  const inst = streams.get(id);
  if (!inst || !inst.active) return;
  if (inst.type === 'server_rtsp') {
    if (inst.previewWs) { inst.previewWs._expectedClose = true; inst.previewWs.close(); inst.previewWs = null; }
    if (inst.eventWs) { inst.eventWs._expectedClose = true; inst.eventWs.close(); inst.eventWs = null; }
    inst._deleteManagedStreamQuietly();
    inst._setBorder('idle');
    await inst._startManagedRtsp();
    toast(`"${inst.name}" source reconnected`, 'success', 2500);
  } else {
    inst._reconnectSourceWs();
  }
}

function refreshStreamsAfterGridChange() {
  streams.forEach(inst => {
    if (!inst.active) return;
    inst._bindCanvas();
    inst._invalidateAnnotatedFrame();
    inst._unstickInferSockets();
  });
}

function updateStreamTotalStats() {
  const active = [...streams.values()].filter(s => s.active);
  const allStreamTiles = [...streams.values()].filter(s => s.tab === 'stream');
  const grid = document.getElementById('stream-grid');
  if (grid) {
    grid.classList.remove('single-camera');
    grid.style.gridTemplateColumns = '';
  }
  const cols = 3;
  const latencies = active
    .map(s => Number(s.lastTotalMs))
    .filter(v => Number.isFinite(v));
  const totalFps = active.reduce((sum, s) => sum + (Number(s.fpsDisplay) || 0), 0);
  const totalBandwidth = active.reduce((sum, s) => sum + (Number(s.bandwidthBps) || 0), 0);
  const totalUpload = active.reduce((sum, s) => sum + (Number(s.uploadBandwidthBps) || 0), 0);
  const totalResult = active.reduce((sum, s) => sum + (Number(s.resultBandwidthBps) || 0), 0);
  const avg = latencies.length ? latencies.reduce((a,b)=>a+b,0) / latencies.length : null;
  const max = latencies.length ? Math.max(...latencies) : null;
  const camsEl = document.getElementById('stream-total-cams');
  const layoutEl = document.getElementById('stream-layout-label');
  const fpsEl = document.getElementById('stream-total-fps');
  const bwEl = document.getElementById('stream-total-bandwidth');
  const upEl = document.getElementById('stream-total-upload');
  const resEl = document.getElementById('stream-total-result');
  const avgEl = document.getElementById('stream-total-avg');
  const maxEl = document.getElementById('stream-total-max');
  if (camsEl) camsEl.textContent = String(active.length);
  if (layoutEl) layoutEl.textContent = `${cols} col fixed`;
  if (fpsEl) fpsEl.textContent = String(totalFps);
  if (bwEl) bwEl.textContent = _fmtBandwidth(totalBandwidth);
  if (upEl) upEl.textContent = _fmtBandwidth(totalUpload);
  if (resEl) resEl.textContent = _fmtBandwidth(totalResult);
  if (avgEl) avgEl.textContent = _fmtMs(avg);
  if (maxEl) maxEl.textContent = _fmtMs(max);
}

function updateStreamEmpty(tab) {
  const gridId = tab === 'rtsp' ? 'rtsp-grid' : 'stream-grid';
  const emptyId = tab === 'rtsp' ? 'rtsp-empty' : 'stream-empty';
  const grid = document.getElementById(gridId);
  const hasStreams = [...streams.values()].some(s => s.tab === tab);
  if (!hasStreams && !document.getElementById(emptyId)) {
    const div = document.createElement('div');
    div.className = 'stream-empty';
    div.id = emptyId;
    div.innerHTML = tab === 'rtsp'
      ? '<div class="empty-icon">📡</div><div>No RTSP streams active</div>'
      : '<div class="empty-icon">📷</div><div>No streams active</div>';
    grid.appendChild(div);
  }
}

function updateStopAllBtn(tab) {
  const btnId = tab + '-stop-all';
  const hasActive = [...streams.values()].some(s => s.tab === tab);
  document.getElementById(btnId).style.display = hasActive ? '' : 'none';
}

function stopAllStreams(tab) {
  [...streams.values()].filter(s => s.tab === tab).forEach(s => s.remove());
}

function applyStreamParams(id) {
  const inst = streams.get(id);
  if (!inst) return;

  const newModels = getSelectedTileModels(id);
  if (!newModels.length) { toast('Select at least 1 model','error'); return; }
  const modelChanged = newModels.join('|') !== inst.models.join('|');
  const newFps = clampFpsInput(document.getElementById(`tf-${id}`), inst.fps, true);
  const newPreviewFps = inst.type === 'server_rtsp'
    ? clampFpsInput(document.getElementById(`tpf-${id}`), inst.previewFps || Math.min(newFps, 10), true)
    : inst.previewFps;
  const nextSourceMaxHeight = inst.type === 'server_rtsp'
    ? parseInt(document.getElementById(`tsh-${id}`)?.value || inst.sourceMaxHeight || 0)
    : inst.sourceMaxHeight;
  const nextRtspBackend = inst.type === 'server_rtsp'
    ? normalizeRtspBackendChoice(document.getElementById(`tbk-${id}`)?.value || inst.rtspBackend || 'auto', true)
    : inst.rtspBackend;

  inst.models  = newModels;
  inst.model   = newModels[0];
  inst.fps     = newFps;
  inst.previewFps = newPreviewFps;
  inst.sourceMaxHeight = Number.isFinite(nextSourceMaxHeight) ? nextSourceMaxHeight : inst.sourceMaxHeight;
  inst.rtspBackend = nextRtspBackend;
  inst.conf    = parseFloat(document.getElementById(`tc-${id}`)?.value)  || inst.conf;
  inst.imgsz   = parseInt(document.getElementById(`ti-${id}`)?.value)    || inst.imgsz;
  inst.classes = document.getElementById(`tcl-${id}`)?.value.trim()      || '';
  const nextOverlayMode = normalizeOverlayMode(document.getElementById(`tom-${id}`)?.value || inst.overlayMode || 'exact');
  const overlayChanged = nextOverlayMode !== inst.overlayMode;
  inst.overlayMode = nextOverlayMode;
  inst.fps = clampFpsValue(inst.fps, 10);
  if (inst.type === 'server_rtsp') {
    inst.previewFps = clampFpsValue(inst.previewFps, 10);
  }
  inst.syncRtspBoxes = inst.type !== 'server_rtsp' && alignedBoxesModeEnabled(inst.overlayMode);
  inst.annotatedPreview = inst.type === 'server_rtsp' && alignedBoxesModeEnabled(inst.overlayMode);
  inst.srcAutoReconnect = !!document.getElementById(`tar-${id}`)?.checked;
  inst.srcStallTimeoutMs = Math.max(2, parseInt(document.getElementById(`tstall-${id}`)?.value) || 5) * 1000;
  const promptModel = inst.models.find(m => modelRequiresPrompts(m));
  if (promptModel && !inst.classes) {
    toast(`"${promptModel}" requires YOLOE prompts`, 'error');
    return;
  }

  // Cập nhật model tag ở footer
  const tag = document.querySelector(`#tile-${id} .tile-model-tag`);
  if (tag) tag.textContent = inst.models.join(', ');

  // Reset stale visual state if model or overlay mode changes.
  if (modelChanged || overlayChanged) {
    inst.catColorMap = {};
    inst.lastAnns = [];
    inst.annsByModel = {};
    inst._invalidateAnnotatedFrame();
    const detsEl = document.getElementById(`tile-dets-${id}`);
    if (detsEl) detsEl.innerHTML = `<div class="tile-no-det">${uiLabel('Waiting')}…</div>`;
  }

  if (inst.type === 'server_rtsp' && inst.managedStreamId) {
    if (overlayChanged) {
      if (inst.previewWs) { inst.previewWs._expectedClose = true; inst.previewWs.close(); inst.previewWs = null; }
      if (inst.eventWs) { inst.eventWs._expectedClose = true; inst.eventWs.close(); inst.eventWs = null; }
      inst._closeGo2Rtc();
      
      // Clear the canvas to erase the last drawn preview JPEG frame pixels
      if (inst.canvas) {
        const ctx = inst.canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, inst.canvas.width, inst.canvas.height);
      }
    }
    const textParams = splitLiveTextParams(inst.models, inst.classes);
    apiFetch(`/streams/${inst.managedStreamId}`, {
      method: 'PATCH',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        models: inst.models,
        expand_ensembles: true,
        classes: textParams.classes,
        prompts: textParams.prompts,
        imgsz: String(inst.imgsz),
        conf: inst.conf,
        fps: inst.fps,
        preview_fps: inst.previewFps,
        max_result_age_ms: 3000,
        live_transport: alignedBoxesModeEnabled(nextOverlayMode) ? 'api_jpeg' : (inst.preferredLiveTransport || 'go2rtc'),
        source_max_height: Number(inst.sourceMaxHeight) || 0,
        backend: inst.rtspBackend,
        annotated_preview: inst.type === 'server_rtsp' && alignedBoxesModeEnabled(nextOverlayMode),
        enable_tracking: inst.enableTracking,
        enable_recording: inst.enableRecording,
      }),
    }).then((snap) => {
      if (snap?.fps != null) inst.fps = Number(snap.fps) || inst.fps;
      if (snap?.preview_fps != null) inst.previewFps = Number(snap.preview_fps) || inst.previewFps;
      if (snap?.source_max_height !== undefined) inst.sourceMaxHeight = Number(snap.source_max_height) || 0;
      if (snap?.live_transport != null) inst.liveTransport = snap.live_transport || 'api_jpeg';
      if (snap?.annotated_preview != null) {
        // Always derive annotatedPreview from type + overlayMode (canonical source of truth).
        // Do NOT blindly accept annotated_preview from the API snapshot — it can be stale
        // (e.g. saved as false when overlayMode was native_exact, now switched to exact).
        // A stale false here would make the client draw JSON boxes on top of server-annotated
        // JPEGs, causing the "dual overlay" visual artifact.
        inst.annotatedPreview = inst.type === 'server_rtsp' && alignedBoxesModeEnabled(nextOverlayMode);
      }

      inst.overlayMode = nextOverlayMode;
      const pf = document.getElementById(`tpf-${id}`);
      if (pf) pf.value = String(inst.previewFps);
      const ff = document.getElementById(`tf-${id}`);
      if (ff) ff.value = String(inst.fps);

      // Reconnect/re-establish transport connections under the new overlayMode
      if (overlayChanged) {
        if (!alignedBoxesModeEnabled(inst.overlayMode) && inst.liveTransport === 'go2rtc' && inst.go2rtcName && inst.go2rtcPublicUrl) {
          inst._connectGo2RtcWebRtc();
        } else {
          inst._connectManagedPreview();
        }
        inst._connectManagedEvents();
      }

      toast(`"${inst.name}" updated · preview ${inst.previewFps} fps · infer ${inst.fps} fps · ${overlayModeLabel(inst.overlayMode)}`, 'success');
    }).catch(() => {});
    return;
  }

  // Ngắt inference WS cũ và kết nối lại với params mới. Do not bump the
  // source generation here; webcam/video/RTSP draw loops should keep running.
  inst._resetInferenceState();
  if (inst.inferWs) { inst.inferWs._expectedClose = true; inst.inferWs.close(); inst.inferWs = null; }
  inst.inferWsList.forEach(w => { w._expectedClose = true; w.close(); });
  inst.inferWsList = [];
  inst._connectInfer();

  toast(`"${inst.name}" → ${inst.models.length} model${inst.models.length>1?'s':''} (conf ${inst.conf})`, 'success');
}
/* ══════════════════════════════════════════════════════════════
   ADD STREAM MODAL
══════════════════════════════════════════════════════════════ */
function openAddStream(tab) {
  addStreamTab = 'stream';
  document.getElementById('add-stream-title').textContent = 'Add Stream';
  onSourceChange({ enumerate: false });
  populateModelSelects();
  document.getElementById('ns-name').value = '';
  document.getElementById('ns-classes').value = '';
  document.getElementById('ns-imgsz').value = '640';
  document.getElementById('ns-conf').value = '0.5';
  document.getElementById('ns-fps').value = String(clampFpsValue(10, 10));
  document.getElementById('ns-preview-fps').value = String(clampFpsValue(10, 10));
  const nsSourceMax = document.getElementById('ns-source-max-height');
  if (nsSourceMax) nsSourceMax.value = '720';
  const nsOverlay = document.getElementById('ns-overlay-mode');
  if (nsOverlay) nsOverlay.value = 'exact';
  const nsAnn = document.getElementById('ns-annotated-preview');
  if (nsAnn) nsAnn.checked = true;
  const nsSync = document.getElementById('ns-box-sync');
  if (nsSync) nsSync.checked = true;
  const nsBackend = document.getElementById('ns-rtsp-backend-select');
  if (nsBackend) {
    nsBackend.innerHTML = rtspBackendOptionsHtml('auto');
    nsBackend.value = 'auto';
  }
  document.getElementById('ns-rtsp-url').value = '';
  document.getElementById('ns-ws-url').value = '';
  document.getElementById('ns-hls-url').value = '';
  applyApiFpsLimit(apiMaxFps);
  nsSelectedFile = null;
  document.getElementById('ns-file-label').textContent = '';
  document.getElementById('add-stream-modal').classList.add('open');
}

function closeAddStream() {
  document.getElementById('add-stream-modal').classList.remove('open');
}

function onSourceChange(opts = {}) {
  const enumerate = opts.enumerate === true;
  const src = document.getElementById('ns-source')?.value || 'rtsp';
  const wsRow = document.getElementById('ns-ws-row');
  if (wsRow) wsRow.style.display = src === 'ws' ? '' : 'none';
  const hlsRow = document.getElementById('ns-hls-row');
  if (hlsRow) hlsRow.style.display = src === 'hls' ? '' : 'none';
  const rtspRow = document.getElementById('ns-rtsp-row');
  if (rtspRow) rtspRow.style.display = src === 'rtsp' ? '' : 'none';
  const backendRow = document.getElementById('ns-rtsp-backend-row');
  if (backendRow) backendRow.style.display = src === 'rtsp' ? '' : 'none';
  const webcamRow = document.getElementById('ns-webcam-row');
  if (webcamRow) webcamRow.style.display = src === 'webcam' ? '' : 'none';
  const fileRow = document.getElementById('ns-file-row');
  if (fileRow) fileRow.style.display = src === 'file' ? '' : 'none';
  const rtspNote = document.getElementById('ns-rtsp-note');
  if (rtspNote) rtspNote.style.display = src === 'rtsp' ? 'block' : 'none';


  const previewWrap = document.getElementById('ns-preview-fps-wrap');
  if (previewWrap) previewWrap.style.display = src === 'rtsp' ? '' : 'none';
  const sourceSizeRow = document.getElementById('ns-source-size-row');
  if (sourceSizeRow) sourceSizeRow.style.display = src === 'rtsp' ? '' : 'none';
  const annWrap = document.getElementById('ns-annotated-preview-wrap');
  if (annWrap) annWrap.style.display = src === 'rtsp' ? '' : 'none';
  const syncWrap = document.getElementById('ns-box-sync-wrap');
  if (syncWrap) syncWrap.style.display = src === 'rtsp' ? 'none' : '';
  if (src === 'webcam') enumerateCameras(enumerate);
  // Show tracking toggle for all stream types
  const trkRow = document.getElementById('ns-tracking-row');
  if (trkRow) trkRow.style.display = '';
  const recRow = document.getElementById('ns-recording-row');
  if (recRow) recRow.style.display = src === 'rtsp' ? '' : 'none';
}

async function _enumerateCamerasInto(sel, requestPermission = false) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    sel.innerHTML = '<option value="">Camera not available (need HTTPS)</option>';
    toast('Camera requires HTTPS or localhost', 'error');
    return;
  }
  if (requestPermission && navigator.mediaDevices.getUserMedia) {
    try {
      // Request permission only for explicit camera refresh/use. This exposes real labels.
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tmp.getTracks().forEach(t => t.stop());
    } catch (e) {
      toast('Camera permission denied or no camera found', 'error');
      sel.innerHTML = '<option value="">Default Camera</option>';
      return;
    }
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');
    if (cams.length && cams.some(d => d.deviceId)) {
      sel.innerHTML = cams.map((d, i) =>
        `<option value="${d.deviceId}">${d.label || 'Camera ' + (i + 1)}</option>`).join('');
    } else {
      // Mobile fallback only — desktop gets default
      const isMobile = /Mobi|Android/i.test(navigator.userAgent);
      sel.innerHTML = isMobile
        ? `<option value="__env__">Back Camera</option>
           <option value="__user__">Front Camera</option>`
        : `<option value="">Default Camera</option>`;
    }
  } catch {
    sel.innerHTML = '<option value="">Default Camera</option>';
  }
}

async function enumerateCameras(requestPermission = false) {
  await _enumerateCamerasInto(document.getElementById('ns-webcam-device'), requestPermission);
}

async function enumerateDetectCameras(requestPermission = true) {
  await _enumerateCamerasInto(document.getElementById('webcam-det-device'), requestPermission);
}

async function probeRtsp() {
  const url = document.getElementById('ns-rtsp-url').value.trim();
  if (!url) { toast('Enter RTSP URL first', 'error'); return; }
  try {
    const data = await apiFetch('/rtsp/probe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, jpeg_quality: 80 }),
    });
    updateRtspBackendUi(data);
    toast(`RTSP OK: ${data.width}×${data.height}${data.source_fps ? ` @ ${data.source_fps}fps` : ''} · ${_rtspBackendText(data)}`, 'success', 8000);
  } catch {}
}

function onFileSelect(e) {
  nsSelectedFile = e.target.files[0];
  if (nsSelectedFile) document.getElementById('ns-file-label').textContent = nsSelectedFile.name;
}

async function addStream() {
  const name  = document.getElementById('ns-name').value.trim() || `Stream ${streamIdCounter+1}`;
  const src     = document.getElementById('ns-source').value;

  let models = getSelectedAddStreamModels();
  if (!models.length) {
    if (src === 'rtsp' && document.getElementById('ns-enable-tracking')?.checked) {
      models = ['person'];
    } else {
      toast('Select at least 1 model','error'); return;
    }
  }

  const classes = document.getElementById('ns-classes').value.trim();
  if (models.some(model => !modelInfoCache[model])) {
    await loadModels();
  }
  const promptModel = models.find(model => modelRequiresPrompts(model));
  if (!classes && promptModel) {
    toast(`"${promptModel}" requires YOLOE prompts (e.g. person,car)`, 'error'); return;
  }
  const imgsz   = parseInt(document.getElementById('ns-imgsz').value) || 640;
  const conf    = parseFloat(document.getElementById('ns-conf').value) || 0.5;
  let fps     = getFpsInputValue('ns-fps', 10);
  let previewFps = getFpsInputValue('ns-preview-fps', Math.min(fps, 10));
  let sourceMaxHeight = parseInt(document.getElementById('ns-source-max-height')?.value || '720');
  let rtspBackend = normalizeRtspBackendChoice(document.getElementById('ns-rtsp-backend-select')?.value || 'auto', true);

  let type, sourceSrc, deviceId;
  if (src === 'webcam') {
    type = 'webcam'; sourceSrc = null;
    deviceId = document.getElementById('ns-webcam-device').value || null;
  } else if (src === 'file') {
    if (!nsSelectedFile) { toast('Choose a video file','error'); return; }
    type = 'file'; sourceSrc = nsSelectedFile;
  } else if (src === 'ws') {
    const url = document.getElementById('ns-ws-url').value.trim();
    if (!url) { toast('Enter WebSocket URL','error'); return; }
    type = 'ws'; sourceSrc = url;
  } else if (src === 'rtsp') {
    const url = document.getElementById('ns-rtsp-url').value.trim();
    if (!url) { toast('Enter RTSP URL','error'); return; }
    if (!/^rtsps?:\/\//i.test(url)) { toast('RTSP URL must start with rtsp:// or rtsps://','error'); return; }
    type = 'server_rtsp';
    sourceSrc = url;
  } else if (src === 'hls') {
    const url = document.getElementById('ns-hls-url').value.trim();
    if (!url) { toast('Enter HLS URL','error'); return; }
    type = 'hls'; sourceSrc = url;
  }

  const overlayMode = normalizeOverlayMode(document.getElementById('ns-overlay-mode')?.value || 'exact');
  fps = clampFpsValue(fps, 10);
  if (type === 'server_rtsp') previewFps = clampFpsValue(previewFps, 10);
  const annotatedPreview = type === 'server_rtsp' && alignedBoxesModeEnabled(overlayMode);
  const syncRtspBoxes = type !== 'server_rtsp' && alignedBoxesModeEnabled(overlayMode);
  const enableTracking = document.getElementById('ns-enable-tracking')?.checked || false;
  const enableRecording = document.getElementById('ns-enable-recording')?.checked || false;
  const id = String(++streamIdCounter);
  const instance = new StreamInstance({
    id, name, type, src: sourceSrc, models, model: models[0], classes, imgsz, conf, fps, previewFps,
    sourceMaxHeight, rtspBackend, annotatedPreview, syncRtspBoxes, overlayMode, tab: addStreamTab, deviceId,
    enableTracking, enableRecording
  });
  streams.set(id, instance);
  instance.start();
  closeAddStream();
  updateStopAllBtn(addStreamTab);
  toast(`Stream "${name}" starting…`, 'success');
}

/* ══════════════════════════════════════════════════════════════
   MODEL UPLOAD
══════════════════════════════════════════════════════════════ */
function handleUploadDragOver(e) { e.preventDefault(); document.getElementById('upload-dropzone').classList.add('drag'); }
document.addEventListener('dragleave', () => document.getElementById('upload-dropzone').classList.remove('drag'));
function handleUploadDrop(e) {
  e.preventDefault();
  document.getElementById('upload-dropzone').classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) setUploadFile(f);
}
function handleModelFileSelect(e) { if (e.target.files[0]) setUploadFile(e.target.files[0]); }
function setUploadFile(f) {
  uploadModelFile = f;
  document.getElementById('upload-filename').textContent = f.name;
  document.getElementById('up-name').placeholder = f.name.replace(/\.(pt|pth)$/, '');
  document.getElementById('upload-btn').disabled = false;
}

async function uploadModel() {
  if (!uploadModelFile) return;
  const btn = document.getElementById('upload-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spin">⟳</span> Uploading…';
  const pw = document.getElementById('upload-progress-wrap');
  const pb = document.getElementById('upload-progress');
  pw.style.display = 'block'; pb.style.width = '10%';
  try {
    const form = new FormData();
    form.append('file', uploadModelFile);
    const name = document.getElementById('up-name').value.trim();
    if (name) form.append('name', name);
    const gpus = getSelectedUploadGpus().join(',');
    if (gpus) form.append('gpus', gpus);
    form.append('imgsz', document.getElementById('up-imgsz').value || '640');
    form.append('overwrite', document.getElementById('up-overwrite').checked ? 'true' : 'false');
    form.append('dynamic', document.getElementById('up-dynamic').checked ? 'true' : 'false');
    form.append('yoloe_dynamic', document.getElementById('up-yoloe').checked ? 'true' : 'false');
    const labels = parseUploadLabels();
    if (labels.length) form.append('labels', labels.join('\n'));
    pb.style.width = '40%';
    HOST = document.getElementById('host-input').value.replace(/\/$/, '');
    const r = await fetch(HOST + '/models/upload', { method:'POST', body:form });
    pb.style.width = '80%';
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
    const data = await r.json();
    pb.style.width = '100%';
    toast(`Model "${data.model}" loaded (${data.type})`,'success');
    setTimeout(() => { pw.style.display='none'; pb.style.width='0%'; }, 800);
    loadModels();
    uploadModelFile = null;
    document.getElementById('upload-filename').textContent = 'Drag .pt / .pth / .onnx here or click';
    document.getElementById('model-file-input').value = '';
    document.getElementById('up-labels').value = '';
  } catch (e) {
    toast('Upload failed: ' + e.message,'error');
    pb.style.width = '0%';
    setTimeout(() => { pw.style.display='none'; }, 400);
  }
  btn.disabled = false; btn.textContent = '⬆ UPLOAD & EXPORT';
}

function modelCard(m) {
  const state = m.state || 'READY';
  const cls = state==='READY'?'badge-green':state==='LOADING'?'badge-yellow':'badge-red';
  const type = [m.type, m.task].filter(Boolean).join(' · ') || m.platform || m.kind || 'onnx';
  const nameAttr = jsAttr(m.name);
  const menuId = `model-menu-${safeDomId(m.name)}`;
  return `<div class="model-card">
    <div class="model-card-top">
      <div>
        <div class="model-card-title">${escHtml(m.name)}</div>
        <div class="model-card-meta">${escHtml(type)}${m.labels_count != null ? ` · ${m.labels_count} labels` : ''}</div>
      </div>
    </div>
    <div class="model-card-footer">
      <span class="badge ${cls} model-status-badge">${state}</span>
      <div class="model-card-actions">
        <button class="btn btn-ghost" style="padding:4px 10px;font-size:10px;" onclick="openModelConfig('${nameAttr}')">${modelIcon('edit')} Config</button>
        <div class="model-menu" id="${menuId}">
          <button class="model-icon-btn" onclick="toggleModelMenu(event, '${menuId}')" title="More actions">${modelIcon('dots')}</button>
          <div class="model-menu-popover">
            <button class="model-menu-item" onclick="closeModelMenus(); modelAction('${nameAttr}', 'info')">${modelIcon('info')} Info</button>
            <button class="model-menu-item" onclick="closeModelMenus(); modelAction('${nameAttr}', 'refresh')">${modelIcon('refresh')} Refresh metadata</button>
            <button class="model-menu-item" onclick="closeModelMenus(); modelAction('${nameAttr}', 'reload')">${modelIcon('reload')} Reload Triton</button>
            <button class="model-menu-item danger" onclick="closeModelMenus(); modelAction('${nameAttr}', 'delete')">${modelIcon('trash')} Delete</button>
          </div>
        </div>
      </div>
    </div>
  </div>`;
}
/* ══════════════════════════════════════════════════════════════
   ENSEMBLE
══════════════════════════════════════════════════════════════ */
function addEnsStep() {
  const sel = document.getElementById('ens-step-select');
  const name = sel.value;
  if (!name) return;
  ensSteps.push({ model:name, version:-1 });
  renderEnsSteps();
}
function renderEnsSteps() {
  document.getElementById('ens-steps').innerHTML = ensSteps.length
    ? ensSteps.map((s,i) => `
      <div class="ens-step">
        <span class="ens-step-label">${i+1}. ${escHtml(s.model)}</span>
        <span class="ens-step-ver">v${s.version===-1?'latest':s.version}</span>
        <button class="ens-step-remove" onclick="removeEnsStep(${i})">✕</button>
      </div>`).join('')
    : '<div style="color:var(--text3);font-family:var(--font-mono);font-size:11px;padding:8px;">No steps added yet</div>';
}
function removeEnsStep(i) { ensSteps.splice(i,1); renderEnsSteps(); }
async function createEnsemble() {
  const name = document.getElementById('ens-name').value.trim();
  if (!name) { toast('Enter ensemble name','error'); return; }
  if (!ensSteps.length) { toast('Add at least one model step','error'); return; }
  try {
    if (editingEnsemble) {
      await apiFetch(`/ensemble/${editingEnsemble}`, {
        method:'PUT', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({steps:ensSteps})
      });
      toast(currentLanguage === 'vi' ? `Ensemble "${editingEnsemble}" đã cập nhật` : `Ensemble "${editingEnsemble}" updated`, 'success');
    } else {
      const data = await apiFetch('/ensemble/create', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name, steps:ensSteps})
      });
      toast(`Ensemble "${data.name||name}" created`,'success');
    }
    cancelEnsembleEdit();
    loadModels();
  } catch {}
}

function editEnsemble(name) {
  const ens = allEnsembles.find(e => e.name === name);
  if (!ens) { toast(currentLanguage === 'vi' ? 'Không tìm thấy ensemble' : 'Ensemble not found', 'error'); return; }
  editingEnsemble = name;
  document.getElementById('ens-name').value = name;
  document.getElementById('ens-name').disabled = true;
  ensSteps = (ens.steps || []).map(s =>
    typeof s === 'string' ? {model:s, version:-1} : {model: s.model||s.name||s, version: s.version??-1}
  );
  renderEnsSteps();
  const title = document.getElementById('ens-panel-title');
  if (title) title.textContent = `Edit: ${name}`;
  const submitBtn = document.getElementById('ens-submit-btn');
  if (submitBtn) submitBtn.textContent = 'UPDATE ENSEMBLE';
  const cancelBtn = document.getElementById('ens-cancel-btn');
  if (cancelBtn) cancelBtn.style.display = '';
  // Scroll to right panel
  document.querySelector('#page-ensemble').scrollIntoView({behavior:'smooth',block:'start'});
  toast(currentLanguage === 'vi' ? `Đang chỉnh sửa "${name}" — sửa steps rồi nhấn UPDATE` : `Editing "${name}" — modify steps then click UPDATE`, 'info');
}

function cancelEnsembleEdit() {
  editingEnsemble = null;
  ensSteps = [];
  renderEnsSteps();
  const nameEl = document.getElementById('ens-name');
  if (nameEl) { nameEl.value = ''; nameEl.disabled = false; }
  const title = document.getElementById('ens-panel-title');
  if (title) title.textContent = 'Create Ensemble';
  const submitBtn = document.getElementById('ens-submit-btn');
  if (submitBtn) submitBtn.textContent = 'CREATE ENSEMBLE';
  const cancelBtn = document.getElementById('ens-cancel-btn');
  if (cancelBtn) cancelBtn.style.display = 'none';
}

async function addStepToEnsemble(ensName) {
  const sel = document.getElementById(`add-step-select-${ensName}`);
  const modelName = sel?.value;
  if (!modelName) { toast(currentLanguage === 'vi' ? 'Chọn mô hình để thêm' : 'Select a model to add', 'error'); return; }
  const ens = allEnsembles.find(e => e.name === ensName);
  if (!ens) return;
  const steps = [...(ens.steps||[]), {model: modelName, version:-1}];
  try {
    await apiFetch(`/ensemble/${ensName}`, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({steps})
    });
    toast(currentLanguage === 'vi' ? `Đã thêm "${modelName}" vào "${ensName}"` : `Added "${modelName}" to "${ensName}"`, 'success');
    loadModels();
  } catch {}
}

async function removeStepFromEnsemble(ensName, stepIdx) {
  const ens = allEnsembles.find(e => e.name === ensName);
  if (!ens) return;
  const steps = (ens.steps || []).filter((_, i) => i !== stepIdx);
  try {
    await apiFetch(`/ensemble/${ensName}`, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({steps})
    });
    toast(currentLanguage === 'vi' ? `Đã xóa step ${stepIdx+1} khỏi "${ensName}"` : `Removed step ${stepIdx+1} from "${ensName}"`, 'success');
    loadModels();
  } catch {}
}

/* ══════════════════════════════════════════════════════════════
   CONFIG / LABELS (improved)
══════════════════════════════════════════════════════════════ */
function setConfigMode(mode) {
  const easy = mode !== 'custom';
  document.getElementById('cfg-easy-pane').style.display = easy ? '' : 'none';
  document.getElementById('cfg-custom-pane').style.display = easy ? 'none' : '';
  document.getElementById('cfg-easy-tab').classList.toggle('active', easy);
  document.getElementById('cfg-custom-tab').classList.toggle('active', !easy);
}

function openModelConfig(name) {
  switchPage('models');
  document.getElementById('cfg-model').value = name;
  setConfigMode('easy');
  loadConfig();
  document.getElementById('cfg-model')?.scrollIntoView({ behavior:'smooth', block:'center' });
}

async function loadConfig() {
  const model = document.getElementById('cfg-model').value;
  if (!model) return;
  document.getElementById('cfg-editor').value = '';
  document.getElementById('labels-list').innerHTML = '<div class="empty-state" style="padding:12px;">Loading…</div>';
  document.getElementById('simple-model-name').value = model;
  let cfg = null;
  let inst = null;
  try {
    cfg = await apiFetch(`/models/${model}/config`);
    document.getElementById('cfg-editor').value = JSON.stringify(cfg, null, 2);
  } catch { document.getElementById('cfg-editor').value = '// Could not load config'; }
  try {
    inst = await apiFetch(`/models/${model}/instances`);
  } catch {}
  renderSimpleConfig(cfg, inst);
  try {
    const lbl = await apiFetch(`/models/${model}/labels`);
    renderLabels(lbl.labels || []);
  } catch { renderLabels([]); }
}

function _instanceGroupsFromPayload(payload) {
  if (!payload) return [];
  return Array.isArray(payload) ? payload : (payload.instance_group || []);
}

function renderSimpleConfig(cfg, instPayload) {
  const groups = _instanceGroupsFromPayload(instPayload);
  const first = groups[0] || {};
  const kind = first.kind || 'KIND_GPU';
  const count = first.count || 1;
  const selected = new Set((groups.flatMap(g => g.gpus || [])).map(String));
  document.getElementById('simple-batch').value = Number.isFinite(Number(cfg?.max_batch_size)) ? String(cfg.max_batch_size) : '0';
  document.getElementById('simple-instance-count').value = String(count);
  document.getElementById('simple-kind').value = kind;
  renderSimpleGpuList(selected);
  toggleSimpleGpuList();
}

function renderSimpleGpuList(selected = null) {
  const wrap = document.getElementById('simple-gpu-list');
  if (!wrap) return;
  const gpus = gpuInfoCache.gpus || [];
  if (!gpus.length) {
    wrap.innerHTML = '<div class="empty-state" style="padding:12px;">No GPU info loaded. Click CONNECT or refresh System GPU info.</div>';
    return;
  }
  const selectedSet = selected || new Set([String(gpuInfoCache.default_gpu ?? gpus[0]?.index ?? 0)]);
  const modelsByGpu = gpuInfoCache.models_by_gpu || {};
  wrap.innerHTML = gpus.map(gpu => {
    const idx = String(gpu.index);
    const assigned = modelsByGpu[idx] || modelsByGpu[gpu.index] || [];
    const vram = gpu.memory_total_mb ? `${(gpu.memory_total_mb / 1024).toFixed(1)} GB VRAM` : 'VRAM unknown';
    return `
      <label class="gpu-option">
        <input type="checkbox" name="simple-gpu" value="${idx}" ${selectedSet.has(idx) ? 'checked' : ''} />
        <div>
          <div class="gpu-option-title">GPU ${idx}${gpu.index === gpuInfoCache.default_gpu ? ' · default' : ''}</div>
          <div class="gpu-option-meta">${escHtml(gpu.name || 'Unknown GPU')} · ${vram}</div>
          <div class="gpu-option-models">${assigned.length ? `Models: ${escHtml(assigned.join(', '))}` : 'No models assigned'}</div>
        </div>
      </label>`;
  }).join('');
}

function toggleSimpleGpuList() {
  const kind = document.getElementById('simple-kind')?.value || 'KIND_GPU';
  const wrap = document.getElementById('simple-gpu-list');
  if (wrap) wrap.style.opacity = kind === 'KIND_GPU' ? '1' : '.45';
  wrap?.querySelectorAll('input[type=checkbox]').forEach(cb => cb.disabled = kind !== 'KIND_GPU');
}

function renderLabels(labels) {
  const list = document.getElementById('labels-list');
  list.innerHTML = '';
  if (!labels.length) {
    list.innerHTML = '<div class="empty-state" style="padding:12px;">No labels — click + Add Label</div>';
    return;
  }
  labels.forEach(l => _appendLabelRow(list, l));
}

function _appendLabelRow(list, val) {
  const idx = list.querySelectorAll('.label-item').length;
  // Remove "empty" placeholder if present
  const empty = list.querySelector('.empty-state');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'label-item';
  div.innerHTML = `
    <div class="label-id">${idx}</div>
    <input class="label-input label-val" value="${escHtml(val)}" placeholder="label name" />
    <button class="label-del-btn" title="Delete" onclick="deleteLabelRow(this)">✕</button>`;
  list.appendChild(div);
}

function addLabelRow() {
  const list = document.getElementById('labels-list');
  _appendLabelRow(list, '');
  // Focus the new input
  const inputs = list.querySelectorAll('.label-val');
  inputs[inputs.length-1]?.focus();
}

function deleteLabelRow(btn) {
  btn.closest('.label-item').remove();
  reIndexLabels();
  const list = document.getElementById('labels-list');
  if (!list.querySelectorAll('.label-item').length) {
    list.innerHTML = '<div class="empty-state" style="padding:12px;">No labels — click + Add Label</div>';
  }
}

function reIndexLabels() {
  document.querySelectorAll('#labels-list .label-item').forEach((item, i) => {
    item.querySelector('.label-id').textContent = i;
  });
}

async function saveLabels() {
  const model = document.getElementById('cfg-model').value;
  if (!model) { toast('Select a model','error'); return; }
  const labels = [...document.querySelectorAll('#labels-list .label-val')]
    .map(i => i.value.trim()).filter(Boolean);
  try {
    await apiFetch(`/models/${model}/labels`, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({labels})
    });
    toast(`${labels.length} label${labels.length!==1?'s':''} saved`,'success');
    reIndexLabels();
  } catch {}
}

async function saveConfig() {
  const model = document.getElementById('cfg-model').value;
  if (!model) { toast('Select a model','error'); return; }
  let cfg;
  try {
    cfg = JSON.parse(document.getElementById('cfg-editor').value);
  } catch (e) {
    toast('Invalid JSON: ' + e.message, 'error');
    return;
  }
  try {
    await apiFetch(`/models/${model}/config`, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(cfg)
    });
    toast('Config saved & model reloaded','success');
  } catch (e) {
    toast(e.message || 'Failed to save config', 'error');
  }
}

async function renameModel() {
  const oldModel = document.getElementById('cfg-model').value;
  const newNameInput = document.getElementById('simple-model-name');
  const newName = (newNameInput ? newNameInput.value : '').trim();

  if (!oldModel) { toast('Select a model first', 'error'); return; }
  if (!newName) { toast('Enter a new model name', 'error'); return; }
  if (newName === oldModel) { toast('Model name is unchanged', 'info'); return; }

  if (!confirm(`Are you sure you want to rename model "${oldModel}" to "${newName}"?\nThis will rename files on disk, update configs & ensembles, and hot-reload Triton.`)) {
    if (newNameInput) newNameInput.value = oldModel;
    return;
  }

  try {
    toast(`Renaming model ${oldModel} → ${newName}…`, 'info');
    await apiFetch(`/models/${oldModel}/rename`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: newName })
    });
    toast(`Model renamed to "${newName}" successfully!`, 'success');

    const select = document.getElementById('cfg-model');
    if (select) {
      let opt = Array.from(select.options).find(o => o.value === newName);
      if (!opt) {
        opt = document.createElement('option');
        opt.value = newName;
        opt.textContent = newName;
        select.appendChild(opt);
      }
      select.value = newName;
    }

    if (typeof loadModels === 'function') {
      await loadModels();
    }
    if (select) {
      select.value = newName;
    }
    await loadConfig();
  } catch (e) {
    if (newNameInput) newNameInput.value = oldModel;
    toast(`Rename failed: ${e.message}`, 'error');
  }
}

async function saveSimpleConfig() {
  const model = document.getElementById('cfg-model').value;
  if (!model) { toast('Select a model','error'); return; }

  // If user changed simple-model-name, trigger rename first
  const newNameInput = document.getElementById('simple-model-name');
  const newName = (newNameInput ? newNameInput.value : '').trim();
  if (newName && newName !== model) {
    await renameModel();
    return;
  }

  const btn = document.getElementById('simple-save-btn');
  const origText = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Reloading…'; }
  try {
    const batch = Math.max(0, parseInt(document.getElementById('simple-batch').value) || 0);
    const count = Math.max(1, parseInt(document.getElementById('simple-instance-count').value) || 1);
    const kind = document.getElementById('simple-kind').value || 'KIND_GPU';
    const gpus = [...document.querySelectorAll('input[name="simple-gpu"]:checked')]
      .map(cb => parseInt(cb.value))
      .filter(n => Number.isFinite(n));
    if (kind === 'KIND_GPU' && !gpus.length) {
      throw new Error('Select at least one GPU, or choose CPU runtime target.');
    }
    // Single PUT /config call with both max_batch_size AND instance_group — one
    // unload+reload cycle.  Sending two sequential requests (config then instances)
    // caused a race where the first reload was racing the second unload, leading to
    // crashes or the GPU assignment being silently dropped.
    const inst = [{ count, kind, ...(kind === 'KIND_GPU' ? { gpus } : {}) }];
    await apiFetch(`/models/${model}/config`, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ max_batch_size: batch, instance_group: inst })
    });
    toast(`GPU assignment saved — model reloaded on GPU ${gpus.join(', ')||kind}`, 'success');
    loadConfig();
    loadGPUs();
  } catch (e) {
    toast('Config error: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = origText; }
  }
}

async function saveInstances() {
  return saveSimpleConfig();
}

/* ══════════════════════════════════════════════════════════════
   TRACKING GALLERY  — /tracked API
══════════════════════════════════════════════════════════════ */
let _trkSearchFile = null;

async function loadTracked() {
  const gallery   = document.getElementById('trk-gallery');
  const countEl   = document.getElementById('trk-count');
  const labelEl   = document.getElementById('trk-gallery-label');
  const clearBtn  = document.getElementById('trk-clear-btn');
  const classFilter = document.getElementById('trk-class-filter');
  const sessionFilter = document.getElementById('trk-session-filter');
  if (!gallery) return;

  gallery.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon" style="animation:spin 1s linear infinite">◎</div>Loading…</div>';

  try {
    const cls = classFilter?.value || '';
    const sess = sessionFilter?.value || 'all';
    
    let url = `/tracked?limit=200`;
    if (cls) url += `&class_name=${encodeURIComponent(cls)}`;
    if (sess && sess !== 'all') url += `&session=${encodeURIComponent(sess)}`;
    
    const data = await apiFetch(url);
    const objects = data.objects || [];
    const sessions = data.sessions || [];

    if (sessionFilter) {
      const currentSess = sessionFilter.value || 'all';
      const sessList = ['admin', ...sessions.filter(s => s !== 'admin')];
      sessionFilter.innerHTML = '<option value="all">All sessions</option>' +
        sessList.map(s => {
          const label = s === 'admin' ? 'Admin session' : s;
          return `<option value="${escAttr(s)}" ${s===currentSess?'selected':''}>${escHtml(label)}</option>`;
        }).join('');
    }

    // Populate class filter from /tracked/classes
    try {
      let clsUrl = '/tracked/classes';
      if (sess && sess !== 'all') clsUrl += `?session=${encodeURIComponent(sess)}`;
      const clsData = await apiFetch(clsUrl);
      const classes = clsData.classes || [];
      const currentVal = classFilter?.value || '';
      if (classFilter) {
        classFilter.innerHTML = '<option value="">All classes</option>' +
          classes.map(c => `<option value="${escAttr(c)}" ${c===currentVal?'selected':''}>${escHtml(c)}</option>`).join('');
      }
      const searchSel = document.getElementById('trk-search-class');
      if (searchSel) {
        searchSel.innerHTML = '<option value="">Any class</option>' +
          classes.map(c => `<option value="${escAttr(c)}">${escHtml(c)}</option>`).join('');
      }
    } catch {}

    if (countEl) countEl.textContent = objects.length ? `${objects.length} objects` : '';
    if (labelEl) labelEl.textContent = objects.length ? `${objects.length} unique` : '';
    if (clearBtn) clearBtn.style.display = objects.length ? '' : 'none';

    if (!objects.length) {
      gallery.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon">◎</div>No tracked objects yet.<br><span style="font-size:10px;color:var(--text3)">Enable tracking on a stream and let it run for a few seconds.</span></div>';
      return;
    }

    gallery.innerHTML = objects.map((obj, idx) => {
      const imgSrc = obj.image_path ? (HOST + obj.image_path) : null;
      const ts = obj.timestamp ? new Date(obj.timestamp + 'Z').toLocaleString() : '';
      return `
        <div class="trk-card flex flex-col justify-between" data-index="${idx}" onclick="trkOnCardClick(event, ${idx}, '${escAttr(obj.global_id)}')" title="${escAttr(obj.global_id)}" style="cursor:pointer;border-radius:var(--radius);overflow:hidden;border:1.5px solid var(--border);background:var(--bg2);transition:all .15s;position:relative;user-select:none;">
          <input type="checkbox" class="trk-card-chk" data-gid="${escAttr(obj.global_id)}" style="position:absolute;top:6px;left:6px;z-index:12;width:16px;height:16px;cursor:pointer;accent-color:#10b981;" onclick="event.stopPropagation();trkOnCardChkChange(event, ${idx})" />
          <button class="absolute top-1.5 right-1.5 w-6 h-6 flex items-center justify-center bg-black/50 hover:bg-red-600 text-white rounded transition-colors border-none" onclick="event.stopPropagation();trkDeleteObject('${escAttr(obj.global_id)}')" title="Delete object" style="z-index:10;padding:0;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="width:11px;height:11px;"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          </button>
          <div style="aspect-ratio:1;background:var(--bg3);display:flex;align-items:center;justify-content:center;overflow:hidden;">
            ${imgSrc
              ? `<img src="${escAttr(imgSrc)}" style="width:100%;height:100%;object-fit:cover;pointer-events:none;" onerror="this.parentElement.innerHTML='<span style=\\'font-size:28px;opacity:.3\\'>◎</span>'" />`
              : '<span style="font-size:28px;opacity:.3">◎</span>'}
          </div>
          <div style="padding:8px 10px;">
            <div style="font-family:var(--font-mono);font-size:10px;font-weight:600;color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(obj.global_id)}</div>
            <div style="font-size:10px;color:var(--text2);margin-top:1px;">${escHtml(obj.class_name || 'object')}</div>
            <div style="font-family:var(--font-mono);font-size:9px;color:var(--text3);margin-top:2px;">${escHtml(ts)}</div>
          </div>
        </div>`;
    }).join('');
    
    trkUpdateSelectionUI();
    setupDragSelection('trk-gallery', '.trk-card', '.trk-card-chk', trkUpdateSelectionUI);
  } catch (e) {
    gallery.innerHTML = `<div class="empty-state" style="grid-column:1/-1;color:var(--red);">Error: ${escHtml(e.message)}</div>`;
  }
}

let _lastTrkIdx = -1;

function trkOnCardClick(e, idx, gid) {
  const cards = document.querySelectorAll('.trk-card');
  const targetCard = cards[idx];
  const targetChk = targetCard ? targetCard.querySelector('.trk-card-chk') : null;
  if (!targetChk) return;

  if (e.shiftKey && _lastTrkIdx !== -1) {
    const start = Math.min(_lastTrkIdx, idx);
    const end = Math.max(_lastTrkIdx, idx);
    const newState = !targetChk.checked;
    for (let i = start; i <= end; i++) {
      const chk = cards[i]?.querySelector('.trk-card-chk');
      if (chk) {
        chk.checked = newState;
        trkUpdateCardStyle(cards[i], newState);
      }
    }
  } else {
    const newState = !targetChk.checked;
    targetChk.checked = newState;
    trkUpdateCardStyle(targetCard, newState);
    _lastTrkIdx = idx;
  }
  trkUpdateSelectionUI();
}

function trkOnCardChkChange(e, idx) {
  const cards = document.querySelectorAll('.trk-card');
  const chk = e.target;
  if (cards[idx]) trkUpdateCardStyle(cards[idx], chk.checked);
  _lastTrkIdx = idx;
  trkUpdateSelectionUI();
}

function trkUpdateCardStyle(card, isSelected) {
  if (!card) return;
  if (isSelected) {
    card.style.borderColor = 'var(--accent, #3b82f6)';
    card.style.boxShadow = '0 0 0 2px var(--accent, #3b82f6)';
  } else {
    card.style.borderColor = 'var(--border)';
    card.style.boxShadow = 'none';
  }
}

function trkUpdateSelectionUI() {
  const chks = document.querySelectorAll('.trk-card-chk');
  const checked = document.querySelectorAll('.trk-card-chk:checked');
  const selBar = document.getElementById('trk-sel-bar');
  const selCount = document.getElementById('trk-sel-count');
  const selectAllBtn = document.getElementById('trk-select-all-btn');
  
  if (selectAllBtn) selectAllBtn.style.display = chks.length > 0 ? '' : 'none';
  if (selBar) selBar.style.display = checked.length > 0 ? 'flex' : 'none';
  if (selCount) selCount.textContent = `${checked.length} selected`;
  
  chks.forEach(chk => {
    const card = chk.closest('.trk-card');
    if (card) trkUpdateCardStyle(card, chk.checked);
  });
}

function trkToggleSelectAll() {
  const chks = document.querySelectorAll('.trk-card-chk');
  const allChecked = Array.from(chks).every(c => c.checked);
  chks.forEach(c => {
    c.checked = !allChecked;
    const card = c.closest('.trk-card');
    if (card) trkUpdateCardStyle(card, !allChecked);
  });
  trkUpdateSelectionUI();
}

function trkClearSelection() {
  const chks = document.querySelectorAll('.trk-card-chk');
  chks.forEach(c => {
    c.checked = false;
    const card = c.closest('.trk-card');
    if (card) trkUpdateCardStyle(card, false);
  });
  trkUpdateSelectionUI();
}

async function trkDeleteSelected() {
  const checked = Array.from(document.querySelectorAll('.trk-card-chk:checked')).map(c => c.getAttribute('data-gid'));
  if (!checked.length) return;
  if (!confirm(`Are you sure you want to delete ${checked.length} selected tracked object(s)?`)) return;
  let count = 0;
  for (const id of checked) {
    try {
      await apiFetch(`/tracked/${encodeURIComponent(id)}`, { method: 'DELETE' });
      count++;
    } catch (e) {
      console.error('Delete object error:', e);
    }
  }
  toast(`Deleted ${count} tracked object(s)`, 'success');
  loadTracked();
}

function trkSelectObject(globalId) {
  const chks = document.querySelectorAll('.trk-card-chk');
  chks.forEach(c => {
    const isTarget = c.getAttribute('data-gid') === globalId;
    c.checked = isTarget;
    const card = c.closest('.trk-card');
    if (card) trkUpdateCardStyle(card, isTarget);
  });
  trkUpdateSelectionUI();
}

async function trkFindSimilar(globalId, imagePath, className) {
  // Highlight card
  document.querySelectorAll('.trk-card').forEach(c => c.style.borderColor = '');
  const selected = [...document.querySelectorAll('.trk-card')].find(c => c.title === globalId);
  if (selected) selected.style.borderColor = 'var(--accent)';

  if (!imagePath) { toast('No image stored for this object', 'error'); return; }

  // Set class filter in search panel
  const searchSel = document.getElementById('trk-search-class');
  if (searchSel && className) searchSel.value = className;

  // Fetch the stored crop image, convert to File, load into search panel, auto-search
  try {
    toast('Loading image for search…', 'info', 1500);
    const imgUrl = HOST + imagePath;
    const resp = await fetch(imgUrl);
    if (!resp.ok) throw new Error(`Image fetch failed: ${resp.status}`);
    const blob = await resp.blob();
    const file = new File([blob], 'crop.jpg', { type: blob.type || 'image/jpeg' });
    _setTrkSearchFile(file);
    // Scroll search panel into view and auto-run
    document.getElementById('trk-search-drop')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    await trkSearch();
  } catch (e) {
    toast('Find similar error: ' + e.message, 'error');
  }
}

async function trkDeleteObject(globalId) {
  if (!confirm(`Delete tracked object "${globalId}"?`)) return;
  try {
    await apiFetch(`/tracked/${encodeURIComponent(globalId)}`, { method: 'DELETE' });
    toast(`Deleted ${globalId}`, 'success');
    loadTracked();
  } catch (e) {
    toast('Delete failed: ' + e.message, 'error');
  }
}

async function clearAllTracked() {
  if (!confirm('Delete ALL tracked objects? This cannot be undone.')) return;
  const data = await apiFetch('/tracked');
  const objects = data.objects || [];
  for (const obj of objects) {
    try { await apiFetch(`/tracked/${encodeURIComponent(obj.global_id)}`, { method: 'DELETE' }); } catch {}
  }
  toast('Cleared all tracked objects', 'success');
  loadTracked();
}

// ── Photo search & Interactive Canvas Cropper ─────────────────────────────────
let _trkRawImage = null;
let _trkCropBox = { x: 0.05, y: 0.05, w: 0.9, h: 0.9 };
let _trkCroppedBlob = null;
let _trkIsDragging = false;
let _trkDragAction = null;
let _trkDragStart = { x: 0, y: 0 };
let _trkBoxStart = { x: 0, y: 0, w: 0, h: 0 };

function openTrkImageSearchModal() {
  const modal = document.getElementById('trk-img-search-modal');
  if (modal) modal.style.display = 'flex';
  
  const classFilter = document.getElementById('trk-class-filter');
  const modalClassSelect = document.getElementById('trk-search-class-modal');
  if (classFilter && modalClassSelect) {
    modalClassSelect.innerHTML = classFilter.innerHTML;
  }
}

function closeTrkImageSearchModal() {
  const modal = document.getElementById('trk-img-search-modal');
  if (modal) modal.style.display = 'none';
}

function handleTrkDropFile(event) {
  const f = event.dataTransfer?.files?.[0];
  if (f) _setTrkSearchFile(f);
}

function handleTrkFileSelect(event) {
  const f = event.target.files?.[0];
  if (f) _setTrkSearchFile(f);
}

function _setTrkSearchFile(f) {
  _trkSearchFile = f;
  const filename = document.getElementById('trk-search-filename');
  if (filename) filename.textContent = `${f.name} (${(f.size / 1024).toFixed(1)} KB)`;

  const reader = new FileReader();
  reader.onload = function(e) {
    const img = new Image();
    img.onload = function() {
      _trkRawImage = img;
      _trkCropBox = { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
      
      const wrap = document.getElementById('trk-search-preview-wrap');
      const ph   = document.getElementById('trk-search-placeholder');
      const btn  = document.getElementById('trk-do-search-btn') || document.getElementById('trk-search-btn');
      
      if (wrap) wrap.style.display = 'flex';
      if (ph)   ph.style.display = 'none';
      if (btn)  btn.disabled = false;

      setTimeout(() => {
        initTrkCanvasCropper();
        updateTrkCropBlob();
      }, 50);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(f);
}

function clearTrkCropFile() {
  _trkSearchFile = null;
  _trkRawImage = null;
  _trkCroppedBlob = null;
  const fileInput = document.getElementById('trk-search-file-input');
  if (fileInput) fileInput.value = '';
  const wrap = document.getElementById('trk-search-preview-wrap');
  const ph   = document.getElementById('trk-search-placeholder');
  const btn  = document.getElementById('trk-do-search-btn') || document.getElementById('trk-search-btn');
  if (wrap) wrap.style.display = 'none';
  if (ph)   ph.style.display = 'flex';
  if (btn)  btn.disabled = true;
}

function initTrkCanvasCropper() {
  const canvas = document.getElementById('trk-crop-canvas');
  if (!canvas || !_trkRawImage) return;

  const maxH = 260;
  const box = document.getElementById('trk-cropper-box');
  const maxW = box ? box.clientWidth || 360 : 360;
  
  const w = _trkRawImage.naturalWidth;
  const h = _trkRawImage.naturalHeight;
  
  const scale = Math.min(maxW / w, maxH / h, 1);
  canvas.width = Math.max(100, Math.round(w * scale));
  canvas.height = Math.max(100, Math.round(h * scale));
  
  renderTrkCropCanvas();
  
  if (!canvas._hasCropEvents) {
    canvas._hasCropEvents = true;
    canvas.addEventListener('pointerdown', handleCropPointerDown);
    canvas.addEventListener('pointermove', handleCropPointerMove);
    window.addEventListener('pointerup', handleCropPointerUp);
  }
}

function renderTrkCropCanvas() {
  const canvas = document.getElementById('trk-crop-canvas');
  if (!canvas || !_trkRawImage) return;
  const ctx = canvas.getContext('2d');
  const cw = canvas.width;
  const ch = canvas.height;
  
  ctx.drawImage(_trkRawImage, 0, 0, cw, ch);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(0, 0, cw, ch);
  
  const bx = Math.round(_trkCropBox.x * cw);
  const by = Math.round(_trkCropBox.y * ch);
  const bw = Math.round(_trkCropBox.w * cw);
  const bh = Math.round(_trkCropBox.h * ch);
  
  ctx.clearRect(bx, by, bw, bh);
  ctx.drawImage(
    _trkRawImage,
    _trkCropBox.x * _trkRawImage.naturalWidth,
    _trkCropBox.y * _trkRawImage.naturalHeight,
    _trkCropBox.w * _trkRawImage.naturalWidth,
    _trkCropBox.h * _trkRawImage.naturalHeight,
    bx, by, bw, bh
  );
  
  ctx.strokeStyle = '#10a37f';
  ctx.lineWidth = 2;
  ctx.strokeRect(bx, by, bw, bh);
  
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(bx + bw / 3, by); ctx.lineTo(bx + bw / 3, by + bh);
  ctx.moveTo(bx + (bw * 2) / 3, by); ctx.lineTo(bx + (bw * 2) / 3, by + bh);
  ctx.moveTo(bx, by + bh / 3); ctx.lineTo(bx + bw, by + bh / 3);
  ctx.moveTo(bx, by + (bh * 2) / 3); ctx.lineTo(bx + bw, by + (bh * 2) / 3);
  ctx.stroke();
  ctx.setLineDash([]);
  
  const handleSize = 8;
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#10a37f';
  ctx.lineWidth = 2;
  const corners = [
    [bx, by],
    [bx + bw, by],
    [bx, by + bh],
    [bx + bw, by + bh]
  ];
  corners.forEach(([cx, cy]) => {
    ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
    ctx.strokeRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
  });

  const origW = Math.round(_trkCropBox.w * _trkRawImage.naturalWidth);
  const origH = Math.round(_trkCropBox.h * _trkRawImage.naturalHeight);
  const info = document.getElementById('trk-crop-info');
  if (info) info.textContent = `Crop Region: ${origW}×${origH}px`;
}

function handleCropPointerDown(e) {
  if (e.preventDefault) e.preventDefault();
  if (e.stopPropagation) e.stopPropagation();
  const canvas = document.getElementById('trk-crop-canvas');
  if (!canvas) return;
  try { canvas.setPointerCapture(e.pointerId); } catch {}
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) / canvas.width;
  const py = (e.clientY - rect.top) / canvas.height;

  _trkIsDragging = true;
  _trkDragStart = { x: px, y: py };
  _trkBoxStart = { ..._trkCropBox };

  const handleDist = 0.08;
  const { x, y, w, h } = _trkCropBox;
  const isFullImage = (x <= 0.02 && y <= 0.02 && w >= 0.96 && h >= 0.96);

  if (!isFullImage && Math.abs(px - x) < handleDist && Math.abs(py - y) < handleDist) _trkDragAction = 'nw';
  else if (!isFullImage && Math.abs(px - (x + w)) < handleDist && Math.abs(py - y) < handleDist) _trkDragAction = 'ne';
  else if (!isFullImage && Math.abs(px - x) < handleDist && Math.abs(py - (y + h)) < handleDist) _trkDragAction = 'sw';
  else if (!isFullImage && Math.abs(px - (x + w)) < handleDist && Math.abs(py - (y + h)) < handleDist) _trkDragAction = 'se';
  else if (!isFullImage && px >= x && px <= x + w && py >= y && py <= y + h) _trkDragAction = 'move';
  else {
    _trkDragAction = 'draw';
    _trkCropBox = { x: Math.max(0, px), y: Math.max(0, py), w: 0.01, h: 0.01 };
  }
}

function handleCropPointerMove(e) {
  if (!_trkIsDragging) return;
  if (e.preventDefault) e.preventDefault();
  if (e.stopPropagation) e.stopPropagation();
  const canvas = document.getElementById('trk-crop-canvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const px = Math.max(0, Math.min(1, (e.clientX - rect.left) / canvas.width));
  const py = Math.max(0, Math.min(1, (e.clientY - rect.top) / canvas.height));
  
  const dx = px - _trkDragStart.x;
  const dy = py - _trkDragStart.y;
  
  if (_trkDragAction === 'move') {
    _trkCropBox.x = Math.max(0, Math.min(1 - _trkBoxStart.w, _trkBoxStart.x + dx));
    _trkCropBox.y = Math.max(0, Math.min(1 - _trkBoxStart.h, _trkBoxStart.y + dy));
  } else if (_trkDragAction === 'draw') {
    _trkCropBox.x = Math.min(_trkDragStart.x, px);
    _trkCropBox.y = Math.min(_trkDragStart.y, py);
    _trkCropBox.w = Math.max(0.02, Math.abs(px - _trkDragStart.x));
    _trkCropBox.h = Math.max(0.02, Math.abs(py - _trkDragStart.y));
  } else {
    let nx = _trkBoxStart.x;
    let ny = _trkBoxStart.y;
    let nw = _trkBoxStart.w;
    let nh = _trkBoxStart.h;

    if (_trkDragAction.includes('w')) {
      const right = _trkBoxStart.x + _trkBoxStart.w;
      nx = Math.min(right - 0.05, Math.max(0, _trkBoxStart.x + dx));
      nw = right - nx;
    }
    if (_trkDragAction.includes('e')) {
      nw = Math.max(0.05, Math.min(1 - _trkBoxStart.x, _trkBoxStart.w + dx));
    }
    if (_trkDragAction.includes('n')) {
      const bottom = _trkBoxStart.y + _trkBoxStart.h;
      ny = Math.min(bottom - 0.05, Math.max(0, _trkBoxStart.y + dy));
      nh = bottom - ny;
    }
    if (_trkDragAction.includes('s')) {
      nh = Math.max(0.05, Math.min(1 - _trkBoxStart.y, _trkBoxStart.h + dy));
    }

    _trkCropBox = { x: nx, y: ny, w: nw, h: nh };
  }

  renderTrkCropCanvas();
}

function handleCropPointerUp(e) {
  if (!_trkIsDragging) return;
  if (e.preventDefault) e.preventDefault();
  if (e.stopPropagation) e.stopPropagation();
  const canvas = document.getElementById('trk-crop-canvas');
  if (canvas && e.pointerId !== undefined) {
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  }
  _trkIsDragging = false;
  _trkDragAction = null;
  updateTrkCropBlob();
}

function getTrkCropBlob() {
  return new Promise((resolve) => {
    if (!_trkRawImage) return resolve(_trkCroppedBlob || null);
    if (_trkCropBox.x <= 0.01 && _trkCropBox.y <= 0.01 && _trkCropBox.w >= 0.98 && _trkCropBox.h >= 0.98) {
      return resolve(null);
    }
    const offCanvas = document.createElement('canvas');
    const cropX = Math.round(_trkCropBox.x * _trkRawImage.naturalWidth);
    const cropY = Math.round(_trkCropBox.y * _trkRawImage.naturalHeight);
    const cropW = Math.max(10, Math.round(_trkCropBox.w * _trkRawImage.naturalWidth));
    const cropH = Math.max(10, Math.round(_trkCropBox.h * _trkRawImage.naturalHeight));
    
    offCanvas.width = cropW;
    offCanvas.height = cropH;
    const ctx = offCanvas.getContext('2d');
    ctx.drawImage(_trkRawImage, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    
    offCanvas.toBlob(blob => {
      _trkCroppedBlob = blob;
      const preview = document.getElementById('trk-search-preview');
      if (preview && blob) {
        if (preview._blobUrl) URL.revokeObjectURL(preview._blobUrl);
        preview._blobUrl = URL.createObjectURL(blob);
        preview.src = preview._blobUrl;
      }
      resolve(blob);
    }, 'image/jpeg', 0.95);
  });
}

function updateTrkCropBlob() {
  getTrkCropBlob();
}

function resetTrkCrop() {
  _trkCropBox = { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
  renderTrkCropCanvas();
  updateTrkCropBlob();
}

function clearTrkCropFile() {
  _trkSearchFile = null;
  _trkCroppedBlob = null;
  _trkRawImage = null;
  const wrap = document.getElementById('trk-search-preview-wrap');
  const ph   = document.getElementById('trk-search-placeholder');
  const btn  = document.getElementById('trk-do-search-btn') || document.getElementById('trk-search-btn');
  const input = document.getElementById('trk-search-file-input') || document.getElementById('trk-search-input');
  
  if (wrap) wrap.style.display = 'none';
  if (ph)   ph.style.display = 'flex';
  if (btn)  btn.disabled = true;
  if (input) input.value = '';
}

async function executeTrkImageSearch() {
  const targetFile = _trkSearchFile || (typeof _trkFile !== 'undefined' ? _trkFile : null);
  if (!targetFile) return;
  
  const btn = document.getElementById('trk-do-search-btn') || document.getElementById('trk-search-btn');
  if (btn) { 
    btn.disabled = true; 
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg> <span>Searching Vector DB…</span>`; 
  }
  
  try {
    const croppedBlob = await getTrkCropBlob();
    const fileToSend = croppedBlob ? new File([croppedBlob], targetFile.name || 'crop.jpg', { type: 'image/jpeg' }) : targetFile;

    const form = new FormData();
    form.append('file', fileToSend);
    const cls = document.getElementById('trk-search-class-modal')?.value || document.getElementById('trk-search-class')?.value || '';
    const threshold = document.getElementById('trk-search-threshold')?.value || '0.70';
    const limit = document.getElementById('trk-search-limit')?.value || '50';
    
    if (cls) form.append('class_name', cls);
    form.append('threshold', threshold);
    form.append('limit', limit);

    const res = await fetch('/api/v1/tracking/search-by-image', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    const hits = data.hits || [];
    closeTrkImageSearchModal();

    const pct = Math.round(parseFloat(threshold) * 100);
    toast(`Found ${hits.length} vector match(es) (similarity ≥ ${pct}%)`, hits.length ? 'success' : 'info');

    // Update gallery header label to indicate search mode with SVG icons
    const label = document.getElementById('trk-gallery-label');
    if (label) {
      label.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="color:#10a37f;font-weight:700;display:flex;align-items:center;gap:4px;">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
            Search Results (${hits.length} matches ≥ ${pct}%)
          </span>
          <button class="btn btn-ghost" onclick="loadTracked()" style="padding:2px 8px;font-size:10px;display:flex;align-items:center;gap:4px;color:var(--danger);">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
            Clear Search
          </button>
        </div>
      `;
    }

    renderTrkSearchResults(hits);
  } catch (e) {
    toast('Image search error: ' + e.message, 'error');
  } finally {
    if (btn) { 
      btn.disabled = false; 
      btn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
        <span>Search Vector DB</span>
      `; 
    }
  }
}

async function trkSearch() {
  await executeTrkImageSearch();
}

function renderTrkSearchResults(hits) {
  const g = document.getElementById('trk-gallery');
  if (!g) return;

  if (!hits.length) {
    g.innerHTML = `
      <div class="empty-state" style="padding:32px 16px;text-align:center;display:flex;flex-direction:column;align-items:center;">
        <div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:50%;margin-bottom:12px;color:var(--primary-variant);">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:4px;">No matching objects found in vector database</div>
        <div style="font-size:11px;color:var(--primary-variant);max-width:320px;">Try lowering the similarity threshold slider (e.g. 50% - 60%) to broaden vector search.</div>
      </div>`;
    return;
  }

  g.innerHTML = hits.map(res => {
    const imgSrc = res.image_path ? res.image_path : null;
    const matchPct = ((res.score || 0) * 100).toFixed(1);
    const ts = res.timestamp ? new Date(res.timestamp + (res.timestamp.endsWith('Z') ? '' : 'Z')).toLocaleString() : '—';
    const gid = res.global_id || 'unknown';
    const className = res.class_name || 'object';

    return `
      <div class="trk-card panel" data-id="${escAttr(gid)}" onclick="trkOpenLightbox('${escAttr(gid)}', '${escAttr(className)}', '', 1)" style="border:1px solid var(--secondary-highlight);background:var(--background-alt);border-radius:8px;padding:0;display:flex;flex-direction:column;gap:0;position:relative;cursor:pointer;overflow:hidden;transition:all 0.2s;" onmouseover="this.style.borderColor='#10a37f';this.style.transform='translateY(-1px)'" onmouseout="this.style.borderColor='var(--secondary-highlight)';this.style.transform='none'">
        
        <!-- Thumbnail Container -->
        <div style="width:100%;aspect-ratio:1/1;background:var(--background);position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;border-bottom:1px solid var(--border);">
          ${imgSrc ? `<img src="${escAttr(imgSrc)}" style="width:100%;height:100%;object-fit:cover;" onerror="this.style.display='none'" />` : `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-primary-variant/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>`}
          
          <!-- Match Score Badge (TOP LEFT) -->
          <div style="position:absolute;top:6px;left:6px;background:rgba(16,163,127,0.92);backdrop-filter:blur(4px);color:#fff;padding:2px 7px;border-radius:4px;font-size:9px;font-weight:700;font-family:var(--font-mono);box-shadow:0 2px 4px rgba(0,0,0,0.5);z-index:10;display:flex;align-items:center;gap:3px;">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" /></svg>
            ${matchPct}% match
          </div>

          <!-- Trashcan button (TOP RIGHT, shows on hover) -->
          <button class="trk-del-btn" onclick="event.stopPropagation();trkDelete('${escAttr(gid)}',this)" style="position:absolute;top:6px;right:6px;width:24px;height:24px;border:none;background:rgba(220,38,38,0.85);backdrop-filter:blur(4px);color:#fff;border-radius:6px;cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:25;box-shadow:0 2px 4px rgba(0,0,0,0.5);" onmouseover="this.style.background='#dc2626';this.style.transform='scale(1.1)'" onmouseout="this.style.background='rgba(220,38,38,0.85)';this.style.transform='none'" title="Delete object">
            <svg xmlns="http://www.w3.org/2000/svg" style="width:12px;height:12px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>

        <!-- Card Body Info -->
        <div style="padding:8px;display:flex;flex-direction:column;gap:3px;">
          <div style="font-family:var(--font-mono);font-size:11px;font-weight:700;color:var(--primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escAttr(gid)}">
            ${escHtml(gid)}
          </div>
          <div style="font-size:10px;color:var(--primary-variant);display:flex;justify-content:space-between;align-items:center;">
            <span style="background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:3px;font-size:9px;text-transform:uppercase;font-weight:700;">${escHtml(className)}</span>
            <span style="font-family:var(--font-mono);font-size:9px;opacity:.7;">${escHtml(res.camera_id || 'cam-01')}</span>
          </div>
          <div style="font-family:var(--font-mono);font-size:9px;color:var(--primary-variant);opacity:.6;margin-top:2px;">
            ${escHtml(ts)}
          </div>
        </div>

      </div>`;
  }).join('');
}

// Auto-load gallery when switching to tracking page
const _origSwitchPage = switchPage;
window.switchPage = function(name) {
  _origSwitchPage(name);
  if (name === 'tracking') loadTracked();
};

// 1. Intercept window.fetch to inject client-side X-API-Key header automatically
const _originalFetch = window.fetch;
window.fetch = function(input, init = {}) {
  const apiKey = localStorage.getItem("api_key") || '';
  if (apiKey) {
    init.headers = init.headers || {};
    if (init.headers instanceof Headers) {
      if (!init.headers.has('X-API-Key')) {
        init.headers.set('X-API-Key', apiKey);
      }
    } else if (Array.isArray(init.headers)) {
      if (!init.headers.some(h => h[0].toLowerCase() === 'x-api-key')) {
        init.headers.push(['X-API-Key', apiKey]);
      }
    } else {
      if (!init.headers['X-API-Key'] && !init.headers['x-api-key']) {
        init.headers['X-API-Key'] = apiKey;
      }
    }
  }
  return _originalFetch(input, init);
};

// 2. Intercept window.WebSocket constructor to append api_key query param for auth
const _OriginalWebSocket = window.WebSocket;
window.WebSocket = function(url, protocols) {
  const apiKey = localStorage.getItem("api_key") || '';
  if (apiKey) {
    try {
      const parsedUrl = new URL(url);
      if (!parsedUrl.searchParams.has('api_key')) {
        parsedUrl.searchParams.set('api_key', apiKey);
      }
      url = parsedUrl.toString();
    } catch (e) {
      if (url.includes('?')) {
        if (!url.includes('api_key=')) {
          url += '&api_key=' + encodeURIComponent(apiKey);
        }
      } else {
        url += '?api_key=' + encodeURIComponent(apiKey);
      }
    }
  }
  return protocols ? new _OriginalWebSocket(url, protocols) : new _OriginalWebSocket(url);
};
window.WebSocket.prototype = _OriginalWebSocket.prototype;

// 3. Initialize API Key Input state from LocalStorage on load
document.addEventListener("DOMContentLoaded", () => {
  const apiKeyInput = document.getElementById("api-key-input");
  if (apiKeyInput) {
    apiKeyInput.value = localStorage.getItem("api_key") || "";
    apiKeyInput.addEventListener("input", (e) => {
      localStorage.setItem("api_key", e.target.value.trim());
    });
  }
});

/* ══════════════════════════════════════════════════════════════
   ADMIN AUTHENTICATION & API KEY MANAGEMENT
   ══════════════════════════════════════════════════════════════ */
let selectedRevokeKeyId = null;

// Auth check on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  checkAdminSession();
  setTimeout(restoreActiveServerStreams, 500);
});

async function checkAdminSession() {
  const savedUser = localStorage.getItem("triton_admin_user");
  const savedRemember = localStorage.getItem("triton_admin_remember") === "true";
  const sessionActive = sessionStorage.getItem("triton_session_active") === "true";
  const rememberCheckbox = document.getElementById("login-remember");
  const userInput = document.getElementById("admin-login-user");
  const passInput = document.getElementById("admin-login-pass");
  
  if (userInput) userInput.value = savedUser || "admin";
  if (passInput) passInput.value = "";
  if (rememberCheckbox) rememberCheckbox.checked = savedRemember;

  // If neither Remember Me nor active session tab exist, force login overlay
  if (!savedRemember && !sessionActive) {
    sessionStorage.removeItem("triton_session_active");
    const afStyle = document.getElementById("anti-flash-style");
    if (afStyle) afStyle.remove();
    const overlay = document.getElementById("admin-login-overlay");
    if (overlay) overlay.style.display = "flex";
    return;
  }

  try {
    const res = await fetch("/api/v1/auth/status");
    const data = await res.json();
    if (data.logged_in) {
      sessionStorage.setItem("triton_session_active", "true");
      const overlay = document.getElementById("admin-login-overlay");
      if (overlay) overlay.style.display = "none";
      // Auto initialize standard NVR UI
      if (typeof initUI === "function") initUI();
      else if (typeof checkHealth === "function") {
        checkHealth();
        loadGPUs();
        loadSystemStatus();
      }
    } else {
      sessionStorage.removeItem("triton_session_active");
      localStorage.removeItem("triton_admin_remember");
      const afStyle = document.getElementById("anti-flash-style");
      if (afStyle) afStyle.remove();
      const overlay = document.getElementById("admin-login-overlay");
      if (overlay) overlay.style.display = "flex";
    }
  } catch (e) {
    console.error("Auth status check failed:", e);
  }
}

async function submitAdminLogin() {
  const userInput = document.getElementById("admin-login-user");
  const passInput = document.getElementById("admin-login-pass");
  const rememberCheckbox = document.getElementById("login-remember");
  const errDiv = document.getElementById("admin-login-error");
  if (errDiv) errDiv.classList.add("hidden");
  
  try {
    const rememberMe = rememberCheckbox ? rememberCheckbox.checked : false;
    const res = await fetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        username: userInput ? userInput.value : "admin",
        password: passInput ? passInput.value : "",
        remember: rememberMe
      })
    });
    const data = await res.json();
    if (res.ok) {
      if (rememberMe) {
        localStorage.setItem("triton_admin_user", userInput ? userInput.value : "admin");
        localStorage.setItem("triton_admin_remember", "true");
      } else {
        localStorage.setItem("triton_admin_user", userInput ? userInput.value : "admin");
        localStorage.removeItem("triton_admin_remember");
      }

      if (passInput) passInput.value = "";
      sessionStorage.setItem("triton_session_active", "true");
      const overlay = document.getElementById("admin-login-overlay");
      if (overlay) overlay.style.display = "none";
      toast("Logged in successfully", "success");
      // Trigger reload to load all components properly
      window.location.reload();
    } else {
      if (errDiv) {
        errDiv.textContent = data.detail || "Authentication failed.";
        errDiv.classList.remove("hidden");
      }
    }
  } catch (e) {
    if (errDiv) {
      errDiv.textContent = "Failed to connect to authentication server.";
      errDiv.classList.remove("hidden");
    }
  }
}

async function logoutAdmin() {
  try {
    await fetch("/api/v1/auth/logout", { method: "POST" });
    sessionStorage.removeItem("triton_session_active");
    localStorage.removeItem("triton_admin_remember");
    toast("Logged out", "info");
    // Trigger full page reload to clear cache/UI state
    window.location.reload();
  } catch (e) {
    toast("Logout failed: " + e.message, "error");
  }
}

function formatRelativeTime(timestampSec) {
  if (!timestampSec) return currentLanguage === 'vi' ? 'Chưa dùng' : 'Never';
  const nowSec = Math.floor(Date.now() / 1000);
  const diff = Math.max(0, nowSec - timestampSec);
  if (diff < 30) return currentLanguage === 'vi' ? 'Vừa xong' : 'Just now';
  if (diff < 60) return `${diff}s ${currentLanguage === 'vi' ? 'trước' : 'ago'}`;
  const mins = Math.floor(diff / 60);
  if (mins < 60) return `${mins}m ${currentLanguage === 'vi' ? 'trước' : 'ago'}`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ${currentLanguage === 'vi' ? 'trước' : 'ago'}`;
  const days = Math.floor(hours / 24);
  return `${days}d ${currentLanguage === 'vi' ? 'trước' : 'ago'}`;
}

async function loadApiKeys() {
  try {
    const res = await fetch("/api/v1/admin/keys");
    const data = await res.json();
    const tbody = document.getElementById("keys-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";
    
    if (!data.keys || data.keys.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="admin-keys-td font-mono" style="text-align:center;padding:24px;color:var(--color-primary-variant);">No active API keys found. Click "+ Create new secret key" to generate one.</td></tr>`;
      
      const activeKeysEl = document.getElementById("stat-active-keys");
      const totalCallsEl = document.getElementById("stat-total-calls");
      const lastActEl = document.getElementById("stat-last-activity");
      if (activeKeysEl) activeKeysEl.innerText = "0";
      if (totalCallsEl) totalCallsEl.innerText = "0";
      if (lastActEl) lastActEl.innerText = currentLanguage === 'vi' ? 'Chưa dùng' : 'Never';
      updateApiKeyStatusUi();
      return;
    }
    
    let totalCalls = 0;
    let mostRecentTime = 0;

    data.keys.forEach(k => {
      const calls = k.usage_count || 0;
      totalCalls += calls;
      if (k.last_used_at && k.last_used_at > mostRecentTime) {
        mostRecentTime = k.last_used_at;
      }

      const rawCreated = k.created_at ? (String(k.created_at).includes('Z') || String(k.created_at).includes('+') ? String(k.created_at) : String(k.created_at).replace(' ', 'T') + 'Z') : null;
      const created = rawCreated ? new Date(rawCreated).toLocaleString() : "N/A";
      const expiresStr = k.expires_at ? new Date(k.expires_at * 1000).toLocaleString() : (currentLanguage === 'vi' ? "Vô hạn" : "Never");
      const timeRemaining = k.expires_at ? formatTimeRemaining(k.expires_at) : "";
      const expiresDisplay = expiresStr + timeRemaining;
      
      const lastUsedDisplay = formatRelativeTime(k.last_used_at);
      const usageDisplay = `<span class="font-mono font-bold text-primary">${calls.toLocaleString()}</span> <span class="text-[10px] text-primary-variant">calls</span>`;

      const scopesBadge = k.scopes.map(s => `<span class="badge-scope">${escHtml(s)}</span>`).join(" ");
      const modelsBadge = k.allowed_models.includes("*") ? '<span class="font-mono text-primary-variant">All (*)</span>' : k.allowed_models.map(m => `<span class="font-mono bg-secondary px-1.5 py-0.5 rounded text-[10px] text-primary border border-secondary-highlight">${escHtml(m)}</span>`).join(" ");
      
      const tr = document.createElement("tr");
      tr.className = "hover:bg-secondary/50 transition-colors";
      tr.innerHTML = `
        <td class="admin-keys-td font-medium text-primary">${escHtml(k.name)}</td>
        <td class="admin-keys-td">
          <div style="display:flex;align-items:center;gap:6px;">
            <span class="key-mask font-mono">${escHtml(k.prefix)}••••••••</span>
            <button class="btn btn-ghost" style="padding:2px 6px;font-size:10px;height:22px;display:flex;align-items:center;justify-content:center;" onclick="revealExistingKey(${k.id}, this)" title="Show Key">👁</button>
          </div>
        </td>
        <td class="admin-keys-td">
          <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-secondary text-primary border border-secondary-highlight">
            <svg class="w-3 h-3 text-primary-variant inline" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
            ${escHtml(k.created_by || 'admin')}
          </span>
        </td>
        <td class="admin-keys-td">${scopesBadge}</td>
        <td class="admin-keys-td">${modelsBadge}</td>
        <td class="admin-keys-td">${usageDisplay}</td>
        <td class="admin-keys-td font-mono text-xs text-primary">${lastUsedDisplay}</td>
        <td class="admin-keys-td text-primary-variant">${escHtml(created)}</td>
        <td class="admin-keys-td text-primary-variant">${escHtml(expiresDisplay)}</td>
        <td class="admin-keys-td" style="text-align:right;">
          <button class="btn btn-ghost text-[10px] text-primary border border-primary/20 px-2 py-1 mr-1 hover:bg-primary/10" onclick="openEditKeyModal(${k.id}, '${escHtml(k.name)}', ${k.expires_at || 'null'})">Edit</button>
          <button class="btn btn-ghost text-[10px] text-danger border border-danger/10 px-2.5 py-1 hover:bg-danger/10" onclick="openRevokeConfirmModal(${k.id}, '${escHtml(k.name)}')">Revoke</button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    const activeKeysEl = document.getElementById("stat-active-keys");
    const totalCallsEl = document.getElementById("stat-total-calls");
    const lastActEl = document.getElementById("stat-last-activity");

    if (activeKeysEl) activeKeysEl.innerText = data.keys.length;
    if (totalCallsEl) totalCallsEl.innerText = totalCalls.toLocaleString();
    if (lastActEl) lastActEl.innerText = formatRelativeTime(mostRecentTime);

    updateApiKeyStatusUi();
  } catch (e) {
    toast("Failed to load API keys: " + e.message, "error");
  }
}

async function refreshApiKeys() {
  const btn  = document.getElementById('keys-refresh-btn');
  const icon = document.getElementById('keys-refresh-icon');
  if (!btn || !icon) { await loadApiKeys(); return; }

  // Spin the icon and disable the button while loading
  btn.disabled = true;
  btn.style.opacity = '0.7';
  icon.style.animation = 'keys-spin 0.7s linear infinite';

  try {
    await loadApiKeys();
    toast('API key stats refreshed', 'success');
  } finally {
    btn.disabled = false;
    btn.style.opacity = '';
    icon.style.animation = '';
  }
}

function openEditKeyModal(id, name, expiresAt) {
  document.getElementById("edit-key-id").value = id;
  document.getElementById("edit-key-name").value = name;
  const select = document.getElementById("edit-key-expiry-select");
  const customWrap = document.getElementById("edit-custom-expiry-wrap");
  if (select) select.value = "keep";
  if (customWrap) customWrap.classList.add("hidden");
  
  const modal = document.getElementById("edit-key-modal");
  if (modal) modal.style.display = "flex";
}

function closeEditKeyModal() {
  const modal = document.getElementById("edit-key-modal");
  if (modal) modal.style.display = "none";
}

function onEditExpirySelectChange() {
  const sel = document.getElementById("edit-key-expiry-select");
  const wrap = document.getElementById("edit-custom-expiry-wrap");
  if (sel && wrap) {
    if (sel.value === "custom") {
      wrap.classList.remove("hidden");
    } else {
      wrap.classList.add("hidden");
    }
  }
}

async function submitEditApiKey() {
  const keyId = document.getElementById("edit-key-id").value;
  const name = document.getElementById("edit-key-name").value.trim();
  const expiryOpt = document.getElementById("edit-key-expiry-select").value;
  
  if (!name) {
    toast(currentLanguage === 'vi' ? 'Vui lòng nhập tên API Key' : 'Please enter a key name', 'warning');
    return;
  }
  
  let expiresAt = undefined;
  
  if (expiryOpt === '0') {
    expiresAt = 0;
  } else if (expiryOpt === 'custom') {
    const customDateVal = document.getElementById("edit-key-expiry-custom").value;
    if (!customDateVal) {
      toast(currentLanguage === 'vi' ? 'Vui lòng chọn ngày hết hạn' : 'Please select a custom expiry date', 'warning');
      return;
    }
    expiresAt = Math.floor(new Date(customDateVal + "T23:59:59").getTime() / 1000);
  } else if (expiryOpt !== 'keep') {
    const days = parseInt(expiryOpt, 10);
    if (!isNaN(days)) {
      expiresAt = Math.floor(Date.now() / 1000) + (days * 86400);
    }
  }
  
  try {
    const body = { name: name };
    if (expiresAt !== undefined) {
      body.expires_at = expiresAt;
    }
    
    const res = await fetch(`/api/v1/admin/keys/${keyId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to update key");
    }
    
    toast(currentLanguage === 'vi' ? 'Đã cập nhật API Key thành công!' : 'API Key updated successfully!', 'success');
    closeEditKeyModal();
    loadApiKeys();
  } catch (e) {
    toast(e.message, 'error');
  }
}

function onExpirySelectChange() {
  const sel = document.getElementById("new-key-expiry-select");
  const wrap = document.getElementById("custom-expiry-wrap");
  if (sel && wrap) {
    if (sel.value === "custom") {
      wrap.classList.remove("hidden");
    } else {
      wrap.classList.add("hidden");
    }
  }
}

function openCreateKeyModal() {
  document.getElementById("new-key-name").value = "";
  document.getElementById("scope-infer").checked = true;
  document.getElementById("scope-read").checked = false;
  document.getElementById("scope-adm").checked = false;
  
  // Set radio opt to all
  const radios = document.getElementsByName("model-access-opt");
  radios.forEach(r => { if (r.value === "all") r.checked = true; });
  onModelAccessOptChange();
  
  // Reset expiration
  const expSelect = document.getElementById("new-key-expiry-select");
  if (expSelect) expSelect.value = "30";
  const customDateInput = document.getElementById("new-key-expiry-custom");
  if (customDateInput) customDateInput.value = "";
  onExpirySelectChange();
  
  // Fetch loaded models to populate checkboxes
  const modelListContainer = document.getElementById("keys-model-list-inner");
  if (modelListContainer) {
    const combined = [...allModels, ...allEnsembles];
    if (combined.length === 0) {
      modelListContainer.innerHTML = `<span class="text-[10px] text-primary-variant p-1">No active models loaded in Triton.</span>`;
    } else {
      modelListContainer.innerHTML = combined.map(m => `
        <label class="flex items-center gap-1.5 text-[11px] text-primary cursor-pointer p-1 rounded hover:bg-secondary">
          <input type="checkbox" class="accent-[#10b981]" value="${escHtml(m.name)}" onchange="_updateKeyModelCount()" />
          <span>${escHtml(m.name)}</span>
        </label>`).join('');
    }
    _updateKeyModelCount();
  }
  
  document.getElementById("create-key-modal").style.display = "flex";
}

function closeCreateKeyModal() {
  document.getElementById("create-key-modal").style.display = "none";
}

function onModelAccessOptChange() {
  const isRestrict = document.querySelector('input[name="model-access-opt"]:checked').value === "restrict";
  const container = document.getElementById("modal-restrict-models-list");
  if (isRestrict) {
    container.classList.remove("hidden");
  } else {
    container.classList.add("hidden");
  }
}

async function submitCreateApiKey() {
  const name = document.getElementById("new-key-name").value.trim();
  if (!name) {
    toast("Key name is required.", "error");
    return;
  }
  
  const scopes = [];
  if (document.getElementById("scope-infer").checked) scopes.push("inference");
  if (document.getElementById("scope-read").checked) scopes.push("data:read");
  if (document.getElementById("scope-adm").checked) scopes.push("admin");
  
  if (scopes.length === 0) {
    toast("Select at least one scope.", "error");
    return;
  }
  
  let allowed_models = ["*"];
  const modelOpt = document.querySelector('input[name="model-access-opt"]:checked').value;
  if (modelOpt === "restrict") {
    allowed_models = [];
    document.querySelectorAll("#modal-restrict-models-list input:checked").forEach(cb => {
      allowed_models.push(cb.value);
    });
    if (allowed_models.length === 0) {
      toast("Select at least one allowed model or select 'All Loaded Models'.", "error");
      return;
    }
  }
  
  const expirySelect = document.getElementById("new-key-expiry-select");
  let expiryDays = null;
  if (expirySelect.value === "custom") {
    const customDateVal = document.getElementById("new-key-expiry-custom").value;
    if (!customDateVal) {
      toast("Please select a custom expiration date.", "error");
      return;
    }
    const selectedDate = new Date(customDateVal);
    const today = new Date();
    selectedDate.setHours(0, 0, 0, 0);
    today.setHours(0, 0, 0, 0);
    const diffTime = selectedDate.getTime() - today.getTime();
    if (diffTime < 0) {
      toast("Expiration date must be in the future.", "error");
      return;
    }
    expiryDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  } else {
    expiryDays = expirySelect.value === "0" ? null : parseInt(expirySelect.value);
  }
  
  try {
    const res = await fetch("/api/v1/admin/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        expires_in_days: expiryDays,
        scopes,
        allowed_models
      })
    });
    const data = await res.json();
    if (res.ok) {
      closeCreateKeyModal();
      plaintextRevealKeyValue = data.api_key;
      isRevealKeyVisible = false;
      const keyEl = document.getElementById("plaintext-reveal-key");
      if (keyEl) {
        keyEl.textContent = "•".repeat(data.api_key.length);
      }
      const btn = document.getElementById("toggle-reveal-key-btn");
      if (btn) {
        btn.textContent = currentLanguage === 'vi' ? 'Hiện' : 'Show';
      }
      document.getElementById("reveal-key-modal").style.display = "flex";
      loadApiKeys();
    } else {
      toast("Failed to create key: " + (data.detail || "Unknown error"), "error");
    }
  } catch (e) {
    toast("Failed to create key: " + e.message, "error");
  }
}

let plaintextRevealKeyValue = "";
let isRevealKeyVisible = false;

function toggleRevealKeyVisibility() {
  const el = document.getElementById("plaintext-reveal-key");
  const btn = document.getElementById("toggle-reveal-key-btn");
  if (!el || !btn) return;
  isRevealKeyVisible = !isRevealKeyVisible;
  if (isRevealKeyVisible) {
    el.textContent = plaintextRevealKeyValue;
    btn.textContent = currentLanguage === 'vi' ? 'Ẩn' : 'Hide';
  } else {
    el.textContent = "•".repeat(plaintextRevealKeyValue.length);
    btn.textContent = currentLanguage === 'vi' ? 'Hiện' : 'Show';
  }
}

function closeRevealKeyModal() {
  document.getElementById("reveal-key-modal").style.display = "none";
  plaintextRevealKeyValue = "";
  isRevealKeyVisible = false;
}

function copyPlaintextKey() {
  if (plaintextRevealKeyValue) {
    navigator.clipboard.writeText(plaintextRevealKeyValue);
    toast("Copied to clipboard!", "success");
  }
}

function openRevokeConfirmModal(id, name) {
  selectedRevokeKeyId = id;
  document.getElementById("revoke-key-display-name").textContent = name;
  document.getElementById("revoke-confirm-modal").style.display = "flex";
}

function closeRevokeConfirmModal() {
  document.getElementById("revoke-confirm-modal").style.display = "none";
  selectedRevokeKeyId = null;
}

async function submitRevokeApiKey() {
  if (!selectedRevokeKeyId) return;
  try {
    const res = await fetch(`/api/v1/admin/keys/${selectedRevokeKeyId}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      toast("API Key revoked successfully.", "success");
      closeRevokeConfirmModal();
      loadApiKeys();
    } else {
      toast("Failed to revoke key: " + (data.detail || "Unknown error"), "error");
    }
  } catch (e) {
    toast("Failed to revoke key: " + e.message, "error");
  }
}

/* ══════════════════════════════════════════════════════════════
   MODEL SELECT OVERLAY MODAL TRIGGERS
   ══════════════════════════════════════════════════════════════ */
function openModelSelectOverlay() {
  const modal = document.getElementById("model-select-overlay");
  if (modal) modal.style.display = "flex";
}

function closeModelSelectOverlay() {
  const modal = document.getElementById("model-select-overlay");
  if (modal) modal.style.display = "none";
  _updateModelCount();
}

function selectAllModelsOverlay(checked) {
  document.querySelectorAll('input[name="d-model-cb"]').forEach(cb => cb.checked = checked);
  _updateModelCount();
}

/* ══════════════════════════════════════════════════════════════
   API KEY FEATURE STATUS BANNER
   ══════════════════════════════════════════════════════════════ */
function updateApiKeyStatusUi() {
  const badge = document.getElementById("api-keys-status-badge");
  const msg = document.getElementById("api-keys-status-msg");
  const banner = document.getElementById("api-keys-status-banner");
  if (!badge || !msg || !banner) return;

  // api-keys-status-banner removed from UI; nothing to update

  if (isRequired) {
    badge.textContent = currentLanguage === 'vi' ? 'ĐANG BẬT' : 'ENABLED';
    badge.className = "px-2 py-0.5 rounded-full font-bold uppercase tracking-wider text-[9px] bg-green-500/10 text-green-500 border border-green-500/20";
    msg.innerHTML = currentLanguage === 'vi' 
      ? 'Yêu cầu khóa API bắt buộc cho mọi truy cập. (Cấu hình qua biến <code>REQUIRE_API_KEY=true</code> trong file .env/docker-compose).'
      : 'API validation is active. All inference requests require keys. (Configured via <code>REQUIRE_API_KEY=true</code> in .env/docker-compose).';
    banner.className = "mb-5 p-3.5 rounded-lg border border-green-500/20 bg-green-500/5 flex items-center justify-between text-xs transition-all duration-200";
  } else {
    badge.textContent = currentLanguage === 'vi' ? 'ĐANG TẮT' : 'DISABLED';
    badge.className = "px-2 py-0.5 rounded-full font-bold uppercase tracking-wider text-[9px] bg-yellow-500/10 text-yellow-500 border border-yellow-500/20";
    msg.innerHTML = currentLanguage === 'vi'
      ? 'Hệ thống đang mở tự do (không yêu cầu khóa). Để bật xác thực, vui lòng cấu hình <code>REQUIRE_API_KEY=true</code> trong file .env hoặc docker-compose.'
      : 'Validation is inactive (public access allowed). To enforce verification, set <code>REQUIRE_API_KEY=true</code> in your .env or docker-compose setup.';
    banner.className = "mb-5 p-3.5 rounded-lg border border-yellow-500/20 bg-yellow-500/5 flex items-center justify-between text-xs transition-all duration-200";
  }
}

/* ══════════════════════════════════════════════════════════════
   API KEY EXPIRE TIME REMAINING
   ══════════════════════════════════════════════════════════════ */
function formatTimeRemaining(expiresAtEpochSeconds) {
  if (!expiresAtEpochSeconds) return "";
  const nowMs = Date.now();
  const expiresMs = expiresAtEpochSeconds * 1000;
  const diffMs = expiresMs - nowMs;
  if (diffMs <= 0) {
    return currentLanguage === 'vi' ? ' (hết hạn)' : ' (expired)';
  }
  
  const totalMin = Math.floor(diffMs / (1000 * 60));
  const min = totalMin % 60;
  
  const totalHr = Math.floor(totalMin / 60);
  const hr = totalHr % 24;
  
  const days = Math.floor(totalHr / 24);
  
  const parts = [];
  if (currentLanguage === 'vi') {
    if (days > 0) parts.push(`${days} ngày`);
    if (hr > 0) parts.push(`${hr} giờ`);
    if (min > 0 || parts.length === 0) parts.push(`${min} phút`);
    return ` (còn ${parts.join(', ')})`;
  } else {
    if (days > 0) parts.push(`${days} day${days > 1 ? 's' : ''}`);
    if (hr > 0) parts.push(`${hr} hour${hr > 1 ? 's' : ''}`);
    if (min > 0 || parts.length === 0) parts.push(`${min} minute${min > 1 ? 's' : ''}`);
    return ` (${parts.join(', ')} left)`;
  }
}

/* ══════════════════════════════════════════════════════════════
   API KEY REVEAL TRIGGER
   ══════════════════════════════════════════════════════════════ */
const revealedKeys = new Map();

async function revealExistingKey(keyId, btnEl) {
  const td = btnEl.closest('td');
  const keySpan = td.querySelector('.key-mask');
  if (!keySpan) return;

  if (revealedKeys.has(keyId)) {
    const prefix = revealedKeys.get(keyId).prefix;
    keySpan.textContent = prefix + "••••••••";
    btnEl.innerHTML = "👁";
    btnEl.title = currentLanguage === 'vi' ? "Hiển thị khóa" : "Show Key";
    revealedKeys.delete(keyId);
    return;
  }

  const warnMsg = currentLanguage === 'vi' 
    ? 'CẢNH BÁO: Hiển thị khóa bí mật có thể làm lộ thông tin bảo mật. Bạn có chắc chắn muốn xem khóa này?'
    : 'WARNING: Revealing the secret key displays sensitive credentials on-screen. Are you sure you want to proceed?';
  
  if (!confirm(warnMsg)) return;

  try {
    const res = await fetch(`/api/v1/admin/keys/${keyId}/reveal`);
    const data = await res.json();
    if (res.ok) {
      const rawKey = data.raw_key;
      const prefix = rawKey.substring(0, 12);
      revealedKeys.set(keyId, { rawKey, prefix });
      
      keySpan.textContent = rawKey;
      btnEl.innerHTML = `<span style="font-weight:bold;opacity:0.8;font-size:12px;">✕</span>`;
      btnEl.title = currentLanguage === 'vi' ? "Ẩn khóa" : "Hide Key";
    } else {
      toast(currentLanguage === 'vi' ? 'Không thể truy xuất khóa: ' + (data.detail || 'Lỗi hệ thống') : 'Failed to retrieve key: ' + (data.detail || 'Server error'), 'error');
    }
  } catch (e) {
    toast(currentLanguage === 'vi' ? 'Lỗi kết nối: ' + e.message : 'Network error: ' + e.message, 'error');
  }
}

/* ══════════════════════════════════════════════════════════════
   ADMIN ACCOUNT MANAGEMENT
   ══════════════════════════════════════════════════════════════ */
let currentEditUsername = null;

async function loadAccounts() {
  try {
    const res = await fetch("/api/v1/admin/accounts");
    const data = await res.json();
    const tbody = document.getElementById("accounts-tbody");
    if (!tbody) return;
    
    tbody.innerHTML = "";
    
    if (!data.accounts || data.accounts.length === 0) {
      tbody.innerHTML = `<tr><td colspan="3" style="text-align:center;padding:20px;color:var(--text3);">No accounts loaded.</td></tr>`;
      return;
    }
    
    data.accounts.forEach(acc => {
      const tr = document.createElement("tr");
      
      const nameCell = acc.is_default 
        ? `<td style="padding-left:16px;font-family:var(--font-mono);font-size:12px;font-weight:600;color:var(--text);">${escHtml(acc.username)} <span class="badge badge-yellow text-[9px] px-1 py-0.5 ml-1 rounded">DEFAULT</span></td>`
        : `<td style="padding-left:16px;font-family:var(--font-mono);font-size:12px;color:var(--text);">${acc.username}</td>`;
      
      const roleBadge = `<span class="badge bg-green-500/10 text-green-500 border border-green-500/20 text-[9px] px-1.5 py-0.5 rounded font-semibold uppercase">${escHtml(acc.role)}</span>`;
      
      let actions = "";
      if (acc.is_default) {
        actions = `<button class="btn btn-ghost" style="padding:2px 8px;font-size:10px;opacity:0.65;" onclick="editAccount('${escHtml(acc.username)}', true)">${currentLanguage === 'vi' ? 'Xem thiết lập' : 'View Config'}</button>`;
      } else {
        actions = `
          <button class="btn btn-ghost" style="padding:2px 8px;font-size:10px;margin-right:4px;" onclick="editAccount('${escHtml(acc.username)}', false)">${currentLanguage === 'vi' ? 'Sửa' : 'Edit'}</button>
          <button class="btn btn-ghost text-danger border border-danger/10 hover:bg-danger/10" style="padding:2px 8px;font-size:10px;" onclick="confirmDeleteAccount('${escHtml(acc.username)}')">${currentLanguage === 'vi' ? 'Xóa' : 'Delete'}</button>
        `;
      }
      
      tr.innerHTML = `
        ${nameCell}
        <td>${roleBadge}</td>
        <td style="text-align: right; padding-right: 16px;">${actions}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error("Failed to load accounts:", e);
    toast("Failed to load accounts: " + e.message, "error");
  }
}

function editAccount(username, isDefault) {
  currentEditUsername = username;
  const formTitle = document.getElementById("account-form-title");
  const usernameInput = document.getElementById("acc-username");
  const passwordInput = document.getElementById("acc-password");
  const envMsg = document.getElementById("acc-env-msg");
  const submitBtn = document.getElementById("acc-submit-btn");
  const cancelBtn = document.getElementById("acc-cancel-btn");
  
  if (formTitle) formTitle.textContent = currentLanguage === 'vi' ? `Chỉnh sửa: ${username}` : `Modify: ${username}`;
  if (usernameInput) {
    usernameInput.value = username;
    usernameInput.disabled = true;
  }
  
  if (isDefault) {
    if (passwordInput) {
      passwordInput.value = "••••••••";
      passwordInput.disabled = true;
      passwordInput.required = false;
    }
    if (envMsg) {
      envMsg.innerHTML = currentLanguage === 'vi'
        ? 'Tài khoản admin mặc định không thể thay đổi tại đây. Để đổi mật khẩu, vui lòng thiết lập biến môi trường <code>ADMIN_PASSWORD</code> trong file .env hoặc docker-compose.yaml của server.'
        : 'The default admin credentials are read-only here. To change the password, update the <code>ADMIN_PASSWORD</code> environment variable inside your .env or docker-compose.yaml server configuration.';
      envMsg.className = "text-[10px] text-yellow-500 bg-yellow-500/5 p-2.5 rounded-lg border border-yellow-500/20 leading-normal block";
    }
    if (submitBtn) submitBtn.style.display = "none";
  } else {
    if (passwordInput) {
      passwordInput.value = "";
      passwordInput.placeholder = "Enter new password";
      passwordInput.disabled = false;
      passwordInput.required = true;
    }
    if (envMsg) {
      envMsg.style.display = "none";
      envMsg.className = "hidden";
    }
    if (submitBtn) {
      submitBtn.style.display = "block";
      submitBtn.textContent = currentLanguage === 'vi' ? "Lưu thay đổi" : "Save Changes";
    }
  }
  
  if (cancelBtn) cancelBtn.classList.remove("hidden");
}

function cancelAccountEdit() {
  currentEditUsername = null;
  const formTitle = document.getElementById("account-form-title");
  const usernameInput = document.getElementById("acc-username");
  const passwordInput = document.getElementById("acc-password");
  const envMsg = document.getElementById("acc-env-msg");
  const submitBtn = document.getElementById("acc-submit-btn");
  const cancelBtn = document.getElementById("acc-cancel-btn");
  
  if (formTitle) formTitle.textContent = currentLanguage === 'vi' ? "Tạo tài khoản mới" : "Create New Account";
  if (usernameInput) {
    usernameInput.value = "";
    usernameInput.disabled = false;
  }
  if (passwordInput) {
    passwordInput.value = "";
    passwordInput.placeholder = "Enter password";
    passwordInput.disabled = false;
    passwordInput.required = true;
  }
  if (envMsg) {
    envMsg.style.display = "none";
    envMsg.className = "hidden";
  }
  if (submitBtn) {
    submitBtn.style.display = "block";
    submitBtn.textContent = currentLanguage === 'vi' ? "Tạo tài khoản" : "Create Account";
  }
  if (cancelBtn) cancelBtn.classList.add("hidden");
}

async function submitAccountAction() {
  const usernameInput = document.getElementById("acc-username");
  const passwordInput = document.getElementById("acc-password");
  if (!usernameInput || !passwordInput) return;
  
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  if (!username || !password) return;
  
  const isEdit = !!currentEditUsername;
  const url = isEdit ? `/api/v1/admin/accounts/${encodeURIComponent(currentEditUsername)}` : "/api/v1/admin/accounts";
  const method = isEdit ? "PUT" : "POST";
  
  try {
    const res = await fetch(url, {
      method: method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (res.ok) {
      toast(isEdit 
        ? (currentLanguage === 'vi' ? 'Cập nhật tài khoản thành công.' : 'Account updated successfully.')
        : (currentLanguage === 'vi' ? 'Tạo tài khoản thành công.' : 'Account created successfully.'),
        'success'
      );
      cancelAccountEdit();
      loadAccounts();
    } else {
      toast(data.detail || "Error processing request", "error");
    }
  } catch (e) {
    toast("Network error: " + e.message, "error");
  }
}

async function confirmDeleteAccount(username) {
  const confirmMsg = currentLanguage === 'vi'
    ? `Bạn có chắc chắn muốn xóa tài khoản "${username}" không? Hành động này không thể hoàn tác.`
    : `Are you sure you want to delete the account "${username}"? This action is permanent.`;
    
  if (!confirm(confirmMsg)) return;
  
  try {
    const res = await fetch(`/api/v1/admin/accounts/${encodeURIComponent(username)}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      toast(currentLanguage === 'vi' ? 'Xóa tài khoản thành công.' : 'Account deleted successfully.', 'success');
      loadAccounts();
    } else {
      toast(data.detail || "Error deleting account", "error");
    }
  } catch (e) {
    toast("Network error: " + e.message, "error");
  }
}

function confirmRemoveStream(id) {
  const inst = streams.get(id);
  if (!inst) return;
  const name = inst.name || id;
  if (typeof showConfirmModal === 'function') {
    showConfirmModal({
      title: 'Delete Stream',
      body: `Are you sure you want to delete stream "<strong>${escHtml(name)}</strong>"?<br><br><span style="font-size:11px;color:var(--text3);">This will stop background inference, recording, and tracking for this camera.</span>`,
      confirmText: 'Delete Stream',
      confirmClass: 'btn-danger',
      onConfirm: () => {
        inst.remove();
      }
    });
  } else {
    if (confirm(`Are you sure you want to delete stream "${name}"?`)) {
      inst.remove();
    }
  }
}

async function restoreActiveServerStreams() {
  try {
    const hostEl = document.getElementById('host-input');
    const HOST = hostEl?.value?.replace(/\/$/, '') || '';
    const res = await fetch(HOST + '/streams');
    if (!res.ok) return;
    const data = await res.json();
    const serverStreams = data.streams || [];
    if (!serverStreams.length) return;

    console.log('[NVR Client] Restoring', serverStreams.length, 'active server stream(s) on page load...');
    for (const s of serverStreams) {
      const existing = [...streams.values()].find(inst => inst.managedStreamId === s.id || inst.src === s.url);
      if (existing) continue;

      const id = String(s.id || ++streamIdCounter);
      const name = s.name || `Camera ${id}`;
      const inst = new StreamInstance({
        id,
        name,
        type: 'server_rtsp',
        src: s.url,
        models: s.requested_models || s.models || [],
        classes: s.classes || '',
        imgsz: s.imgsz || 640,
        conf: s.conf ?? 0.5,
        fps: s.fps || 30,
        previewFps: s.preview_fps || 10,
        sourceMaxHeight: s.source_max_height || 720,
        rtspBackend: s.backend || 'auto',
        overlayMode: 'native_exact',
        tab: 'live',
        enableTracking: !!s.tracking_enabled,
        enableRecording: !!s.recording_enabled
      });

      inst.managedStreamId = s.id;
      inst.liveTransport = s.live_transport || 'go2rtc';
      inst.go2rtcName = s.go2rtc_name || null;
      inst.go2rtcPublicUrl = s.go2rtc_public_url || null;
      inst.active = true;
      inst.generation++;

      // Render tile to DOM
      inst._renderTile();
      streams.set(inst.id, inst);

      // Connect video stream preview & inference feeds
      inst.annotatedPreview = inst.type === 'server_rtsp' && alignedBoxesModeEnabled(inst.overlayMode);
      if (!alignedBoxesModeEnabled(inst.overlayMode) && inst.liveTransport === 'go2rtc' && inst.go2rtcName && inst.go2rtcPublicUrl) {
        inst._connectGo2RtcWebRtc();
      } else {
        inst._connectManagedPreview();
      }
      inst._connectManagedEvents();
      inst._setBorder('live');
    }
    if (typeof updateStreamTotalStats === 'function') updateStreamTotalStats();
  } catch (e) {
    console.warn('[NVR Client] Error restoring server streams:', e);
  }
}

