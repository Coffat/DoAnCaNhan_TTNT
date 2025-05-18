# Báo cáo Project: Giải Bài Toán 8-Puzzle Với Các Thuật Toán Tìm Kiếm và Học Tăng Cường
**Sinh viên thực hiện: Vũ Toàn Thắng - 23110329**
![GIF](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/DoAnCaNhan1.gif?raw=true)

## Mục lục
1.  [Giới thiệu](#giới-thiệu)
2.  [Tổng Quan Tính Năng](#tổng-quan-tính-năng)
3.  [Kiến Trúc Thuật Toán](#kiến-trúc-thuật-toán)
    1.  [Nhóm Thuật Toán Tìm Kiếm Không Có Thông Tin (Uninformed Search)](#nhóm-thuật-toán-tìm-kiếm-không-có-thông-tin-uninformed-search)
    2.  [Nhóm Thuật Toán Tìm Kiếm Có Thông Tin (Informed Search)](#nhóm-thuật-toán-tìm-kiếm-có-thông-tin-informed-search)
    3.  [Nhóm Thuật Toán Tìm Kiếm Cục Bộ (Local Search)](#nhóm-thuật-toán-tìm-kiếm-cục-bộ-local-search)
    4.  [Nhóm Thuật Toán Cho Môi Trường Phức Tạp và Không Xác Định](#nhóm-thuật-toán-cho-môi-trường-phức-tạp-và-không-xác-định)
    5.  [Nhóm Thuật Toán Giải Quyết Vấn Đề Thỏa Mãn Ràng Buộc (Constraint Satisfaction)](#nhóm-thuật-toán-giải-quyết-vấn-đề-thỏa-mãn-ràng-buộc-constraint-satisfaction)
    6.  [Nhóm Thuật Toán Học Tăng Cường (Reinforcement Learning)](#nhóm-thuật-toán-học-tăng-cường-reinforcement-learning)
4.  [Hướng Dẫn Vận Hành Chi Tiết](#hướng-dẫn-vận-hành-chi-tiết)
    1.  [Khởi Động và Giao Diện Chính](#khởi-động-và-giao-diện-chính)
    2.  [Lựa Chọn và Cấu Hình Thuật Toán](#lựa-chọn-và-cấu-hình-thuật-toán)
    3.  [Tương Tác Với Bàn Cờ 8-Puzzle](#tương-tác-với-bàn-cờ-8-puzzle)
    4.  [Thực Hiện Giải Puzzle](#thực-hiện-giải-puzzle)
    5.  [Vận Hành Chuyên Sâu Với Q-Learning](#vận-hành-chuyên-sâu-với-q-learning)
    6.  [Khám Phá Môi Trường Không Xác Định](#khám-phá-môi-trường-không-xác-định)
    7.  [Phân Tích và Đánh Giá Hiệu Suất Thuật Toán](#phân-tích-và-đánh-giá-hiệu-suất-thuật-toán)
    8.  [Khai Thác và Lưu Trữ Kết Quả](#khai-thác-và-lưu-trữ-kết-quả)
5.  [Phân Tích Hiệu Suất và Kết Quả Đánh Giá](#phân-tích-hiệu-suất-và-kết-quả-đánh-giá)
6.  [Hướng Dẫn Cài Đặt và Triển Khai](#hướng-dẫn-cài-đặt-và-triển-khai)

## Giới thiệu

**Dự án** là một ứng dụng phần mềm tiên tiến, được phát triển bằng Python, chuyên dùng để giải quyết và phân tích sâu bài toán 8-Puzzle kinh điển. Dự án này nổi bật với giao diện người dùng đồ họa (GUI) được xây dựng trên nền tảng PyQt5, mang đến một môi trường tương tác trực quan và hiệu quả. Không chỉ dừng lại ở việc giải puzzle, ứng dụng còn tích hợp một bộ sưu tập phong phú các thuật toán tìm kiếm Trí tuệ Nhân tạo (AI), cho phép người dùng không chỉ trực quan hóa quá trình giải quyết vấn đề mà còn so sánh và đối chiếu hiệu suất giữa các phương pháp tiếp cận khác nhau. Điểm đặc biệt của dự án là khả năng khám phá các lĩnh vực tìm kiếm nâng cao, bao gồm học tăng cường và các thuật toán được thiết kế cho môi trường không xác định.


## Tổng Quan Tính Năng

Ứng dụng được trang bị một loạt các chức năng mạnh mẽ, đáp ứng nhu cầu đa dạng của người dùng:

Một **giao diện đồ họa hiện đại và tinh tế**, được phát triển với thư viện PyQt5 và tuân theo các nguyên tắc của Material Design, đảm bảo trải nghiệm người dùng tối ưu.

Khả năng **triển khai hơn 20 thuật toán tìm kiếm** đa dạng, được phân chia thành các nhóm chuyên biệt, từ các phương pháp cơ bản đến nâng cao.

Tính năng **trực quan hóa động quá trình giải quyết puzzle**, cho phép theo dõi từng bước di chuyển của các ô số, được hỗ trợ bởi hiệu ứng hoạt họa mượt mà.

Cung cấp **công cụ tùy chỉnh bàn cờ puzzle linh hoạt**, bao gồm việc đặt lại về trạng thái ban đầu, khởi tạo một puzzle ngẫu nhiên (đảm bảo có lời giải), và tính năng dự kiến cho phép người dùng nhập một trạng thái puzzle cụ thể.

Chế độ **tự động trình diễn lời giải**, giúp người dùng dễ dàng theo dõi toàn bộ chuỗi hành động dẫn đến trạng thái đích.

Một **mô-đun chuyên dụng dành cho thuật toán Q-Learning**, bao gồm một cửa sổ giao diện riêng biệt cho phép huấn luyện agent, tinh chỉnh các siêu tham số, giám sát tiến trình học và hiển thị trực quan các giá trị Q (Q-values).

Khả năng **mô phỏng và tương tác với môi trường 8-Puzzle không xác định**. Một cửa sổ chuyên biệt được thiết kế để người dùng có thể khám phá và áp dụng các thuật toán như AND/OR Graph Search, Partially Observable Search, và Belief State Search trong các điều kiện môi trường có tính không chắc chắn, ví dụ như thông tin quan sát được bị hạn chế hoặc kết quả của hành động mang tính ngẫu nhiên.

Một hệ thống **đánh giá và so sánh hiệu suất thuật toán toàn diện**:
-   Thực hiện các thử nghiệm đánh giá trên một tập hợp các puzzle được khởi tạo ngẫu nhiên, áp dụng cho các thuật toán hoặc nhóm thuật toán do người dùng lựa chọn.
-   Trình bày kết quả đánh giá dưới dạng bảng so sánh chi tiết, bao gồm các số liệu quan trọng như thời gian thực thi trung bình, độ dài trung bình của đường đi, tỷ lệ giải thành công, và ước tính số lượng nút đã mở rộng trong quá trình tìm kiếm.
-   Minh họa kết quả bằng các biểu đồ cột trực quan, giúp dễ dàng so sánh hiệu suất về thời gian, độ dài đường đi và tỷ lệ thành công.
-   Cung cấp một giao diện dạng thẻ (card-based) hiện đại để so sánh hiệu suất chuyên sâu theo từng nhóm thuật toán.
![Ảnh 1](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/2-h%C3%ACnh%20%E1%BA%A3nh-0.jpg?raw=true)
![Ảnh 2](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/2-h%C3%ACnh%20%E1%BA%A3nh-1.jpg?raw=true)
![Ảnh 3](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/2-h%C3%ACnh%20%E1%BA%A3nh-2.jpg?raw=true)
![Ảnh 4](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/2-h%C3%ACnh%20%E1%BA%A3nh-3.jpg?raw=true)
![Ảnh 5](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/2-h%C3%ACnh%20%E1%BA%A3nh-4.jpg?raw=true)
![Ảnh 6](https://github.com/Coffat/DoAnCaNhan_TTNT/blob/main/image/2-h%C3%ACnh%20%E1%BA%A3nh-5.jpg?raw=true)

Chức năng **xuất báo cáo và kết quả linh hoạt**:
-   Lưu trữ chi tiết lời giải của một puzzle (chuỗi các bước di chuyển) ra một tệp văn bản định dạng `.txt`.
-   Tạo và xuất một báo cáo đánh giá thuật toán hoàn chỉnh, bao gồm cả bảng số liệu và các biểu đồ phân tích, dưới định dạng PDF chuyên nghiệp.

Hệ thống **ghi nhật ký (logging) chi tiết**, theo dõi và lưu lại tất cả các hành động quan trọng của người dùng và trạng thái hoạt động của ứng dụng, hỗ trợ quá trình gỡ lỗi và phân tích.

## Kiến Trúc Thuật Toán

Dự án tích hợp một thư viện thuật toán tìm kiếm đa dạng và phong phú, được tổ chức một cách khoa học thành các nhóm dựa trên đặc tính và phương pháp tiếp cận của chúng.

### Nhóm Thuật Toán Tìm Kiếm Không Có Thông Tin (Uninformed Search)
Nhóm này bao gồm các thuật toán duyệt không gian trạng thái mà không dựa vào bất kỳ thông tin ước lượng nào về khoảng cách hay chi phí để đạt đến trạng thái đích (heuristic). Các thuật toán này hoạt động dựa trên cấu trúc của đồ thị trạng thái.

**Thuật toán BFS (Breadth-First Search - Tìm kiếm theo chiều rộng):**
Phương pháp này khám phá tất cả các trạng thái ở cùng một độ sâu trước khi chuyển sang các trạng thái ở độ sâu tiếp theo. Một ưu điểm nổi bật của BFS là nó đảm bảo tìm ra đường đi ngắn nhất (về số bước) nếu lời giải tồn tại. Tuy nhiên, BFS thường đòi hỏi một lượng lớn bộ nhớ để lưu trữ các trạng thái chờ duyệt.

**Thuật toán DFS (Depth-First Search - Tìm kiếm theo chiều sâu):**
DFS ưu tiên đi sâu vào một nhánh của cây tìm kiếm cho đến khi không thể đi tiếp hoặc đạt đến giới hạn độ sâu đã định trước, sau đó mới quay lui để khám phá các nhánh khác. DFS thường tiết kiệm bộ nhớ hơn BFS nhưng không đảm bảo tìm ra lời giải ngắn nhất và có thể bị lạc vào các nhánh vô hạn nếu không có cơ chế giới hạn độ sâu.

**Thuật toán UCS (Uniform Cost Search - Tìm kiếm chi phí đồng nhất):**
UCS mở rộng các nút dựa trên chi phí tích lũy thấp nhất từ trạng thái bắt đầu (g(n)). Nó đảm bảo tìm ra đường đi có tổng chi phí thấp nhất, miễn là chi phí của mỗi bước là không âm. Trong bài toán 8-Puzzle với chi phí mỗi bước di chuyển là đồng nhất (ví dụ, bằng 1), UCS hoạt động tương tự như BFS.

**Thuật toán IDS (Iterative Deepening Search - Tìm kiếm sâu dần):**
IDS là một giải pháp kết hợp thông minh giữa BFS và DFS. Nó thực hiện một loạt các tìm kiếm DFS với giới hạn độ sâu tăng dần. Điều này cho phép IDS tận dụng được ưu điểm về tính đầy đủ và tối ưu của BFS trong khi vẫn duy trì được yêu cầu bộ nhớ thấp tương tự DFS.

### Nhóm Thuật Toán Tìm Kiếm Có Thông Tin (Informed Search)
Các thuật toán thuộc nhóm này sử dụng một hàm heuristic (h(n)) để ước lượng chi phí hoặc "sự hứa hẹn" từ một trạng thái hiện tại đến trạng thái đích. Thông tin heuristic này giúp hướng dẫn quá trình tìm kiếm một cách hiệu quả hơn, tập trung vào các hướng có khả năng dẫn đến lời giải nhanh hơn.

**Thuật toán Greedy Best-First Search (Tìm kiếm Tham lam theo Hướng Tốt nhất):**
Phương pháp này luôn chọn mở rộng nút có vẻ tốt nhất tại thời điểm hiện tại, dựa hoàn toàn vào giá trị của hàm heuristic h(n). Greedy Search thường tìm đến lời giải rất nhanh nhưng không có gì đảm bảo rằng lời giải đó sẽ là tối ưu.

**Thuật toán A\* Search (Tìm kiếm A-sao):**
A\* là một trong những thuật toán tìm kiếm có thông tin mạnh mẽ và phổ biến nhất. Nó đánh giá các nút dựa trên hàm f(n) = g(n) + h(n), trong đó g(n) là chi phí thực tế từ trạng thái bắt đầu đến nút n, và h(n) là chi phí ước tính từ nút n đến trạng thái đích. Nếu hàm heuristic h(n) là "chấp nhận được" (admissible - không bao giờ đánh giá quá cao chi phí thực tế) và "nhất quán" (consistent), A\* đảm bảo tìm ra đường đi tối ưu và là một thuật toán đầy đủ.

**Thuật toán IDA\* Search (Iterative Deepening A\* - Tìm kiếm A-sao Sâu dần):**
IDA\* là một biến thể của A\* sử dụng kỹ thuật tìm kiếm sâu dần. Thay vì lưu trữ tất cả các nút đã mở rộng trong bộ nhớ (như A\* truyền thống), IDA\* thực hiện một loạt các tìm kiếm theo chiều sâu. Mỗi lượt tìm kiếm DFS được giới hạn bởi một ngưỡng chi phí f(n) tăng dần. Điều này giúp IDA\* tiết kiệm bộ nhớ đáng kể so với A\*, trong khi vẫn duy trì được tính đầy đủ và tối ưu nếu heuristic là chấp nhận được.

### Nhóm Thuật Toán Tìm Kiếm Cục Bộ (Local Search)
Các thuật toán tìm kiếm cục bộ hoạt động trên một trạng thái hiện tại (hoặc một tập nhỏ các trạng thái) và cố gắng cải thiện nó bằng cách thực hiện các thay đổi nhỏ, cục bộ. Chúng thường không quan tâm đến đường đi đã qua mà chỉ tập trung vào trạng thái hiện tại và các lân cận của nó.

**Thuật toán Hill Climbing (Leo đồi):**
Đây là một phương pháp tìm kiếm cục bộ đơn giản. Từ trạng thái hiện tại, nó di chuyển đến một trạng thái lân cận tốt hơn (ví dụ, có giá trị heuristic thấp hơn). Hill Climbing dễ triển khai nhưng có nhược điểm là dễ bị mắc kẹt tại các "cực đại cục bộ" (local maxima) – những điểm tốt hơn tất cả các lân cận nhưng không phải là lời giải toàn cục tốt nhất. Phiên bản được triển khai trong dự án này bao gồm các cải tiến như khả năng thử di chuyển ngẫu nhiên khi bị kẹt để cố gắng thoát khỏi cực đại cục bộ.

**Thuật toán Steepest Ascent Hill Climbing (Leo đồi Dốc nhất):**
Biến thể này của Hill Climbing đánh giá tất cả các trạng thái lân cận và chọn di chuyển đến trạng thái tốt nhất trong số đó. Mặc dù có vẻ "tham lam" hơn, nó vẫn có thể bị mắc kẹt ở các cực đại cục bộ.

**Thuật toán Stochastic Hill Climbing / Random Restart Hill Climbing:**
Để giải quyết vấn đề cực đại cục bộ, Stochastic Hill Climbing có thể chọn một lân cận tốt hơn một cách ngẫu nhiên (thay vì luôn chọn cái tốt nhất). Một chiến lược hiệu quả hơn là Random Restart Hill Climbing (được triển khai với tên `hill_climbing_random_restart`), trong đó thuật toán leo đồi được thực hiện nhiều lần, mỗi lần bắt đầu từ một trạng thái khởi tạo ngẫu nhiên. Lời giải tốt nhất trong tất cả các lần chạy sẽ được giữ lại.

**Thuật toán Local Beam Search (Tìm kiếm Tia Cục bộ):**
Thay vì chỉ theo dõi một trạng thái hiện tại, Local Beam Search duy trì một "chùm" (beam) gồm k trạng thái tốt nhất. Ở mỗi bước, nó tạo ra tất cả các trạng thái kế tiếp của k trạng thái này và sau đó chọn ra k trạng thái tốt nhất từ tập hợp các trạng thái kế tiếp đó để hình thành chùm cho bước tiếp theo. Điều này giúp khám phá không gian tìm kiếm một cách song song hơn và giảm nguy cơ bị kẹt sớm.

**Thuật toán Simulated Annealing (Mô phỏng Luyện kim):**
Lấy cảm hứng từ quá trình luyện kim trong vật lý, Simulated Annealing cho phép di chuyển đến các trạng thái xấu hơn với một xác suất nhất định. Xác suất này phụ thuộc vào mức độ "xấu" của trạng thái mới và một tham số "nhiệt độ" (temperature) giảm dần theo thời gian. Khi nhiệt độ cao, thuật toán có xu hướng khám phá nhiều hơn; khi nhiệt độ giảm, nó trở nên "tham lam" hơn và tập trung vào việc cải thiện. Khả năng chấp nhận các bước đi xấu giúp Simulated Annealing thoát khỏi các cực đại cục bộ.

**Thuật toán Genetic Algorithm (Thuật toán Di truyền):**
Genetic Algorithm mô phỏng các nguyên lý của quá trình tiến hóa tự nhiên. Nó hoạt động trên một "quần thể" (population) các lời giải tiềm năng (thường được mã hóa dưới dạng chuỗi hoặc "nhiễm sắc thể"). Các toán tử di truyền như "chọn lọc" (selection), "lai ghép" (crossover), và "đột biến" (mutation) được áp dụng để tạo ra các thế hệ lời giải mới, với hy vọng rằng các lời giải tốt hơn sẽ xuất hiện theo thời gian.

### Nhóm Thuật Toán Cho Môi Trường Phức Tạp và Không Xác Định
Nhóm này bao gồm các thuật toán được thiết kế để đối phó với các tình huống mà thông tin về môi trường có thể không đầy đủ, hoặc kết quả của các hành động không hoàn toàn chắc chắn.

**Thuật toán AND/OR Graph Search (Tìm kiếm Đồ thị AND/OR):**
Phương pháp này phù hợp cho các bài toán có thể được phân rã thành các bài toán con. Trong đồ thị AND/OR, một nút OR đại diện cho việc lựa chọn một hành động (chỉ cần một hành động thành công), trong khi một nút AND đại diện cho việc phải xử lý thành công tất cả các kết quả có thể xảy ra của một hành động không xác định. Phiên bản cải tiến (`andor_graph_search_improved`) trong dự án sử dụng heuristic, kỹ thuật tìm kiếm sâu dần (iterative deepening) và bộ đệm (cache) để tối ưu hóa hiệu suất. Đặc biệt, `andor_graph_search_non_deterministic` là phiên bản được tùy chỉnh cho cửa sổ mô phỏng môi trường không xác định, có khả năng làm việc với các "trạng thái niềm tin" (belief states).

**Thuật toán Fixed Partially Observable Search (Tìm kiếm Quan sát Được Một Phần Cố Định):**
Trong kịch bản này, agent chỉ có thể quan sát được một tập hợp cố định các ô trên bàn cờ 8-Puzzle. Các quyết định phải được đưa ra dựa trên thông tin hạn chế này. Thuật toán thường phải duy trì một tập hợp các "trạng thái niềm tin" – những trạng thái có thể là trạng thái thực sự của môi trường, tương thích với những gì agent quan sát được. Phiên bản cải tiến (`improved_partially_observable`) kết hợp A\* để tìm kiếm trên trạng thái niềm tin có khả năng xảy ra cao nhất, được thiết kế cho cửa sổ mô phỏng môi trường không xác định.

**Thuật toán Belief State Search (Tìm kiếm Trên Không Gian Trạng Thái Niềm Tin):**
Tương tự như Fixed Partially Observable Search, thuật toán này hoạt động trực tiếp trên không gian của các trạng thái niềm tin. Nó cập nhật và tìm kiếm trên tập hợp các trạng thái có thể này. Phiên bản cải tiến (`improved_belief_state_search`) trong dự án sử dụng các khái niệm từ Quy hoạch Động Quyết định (Markov Decision Process - MDP) để đưa ra quyết định tối ưu hơn trong môi trường không chắc chắn, được tích hợp vào cửa sổ mô phỏng chuyên dụng.

### Nhóm Thuật Toán Giải Quyết Vấn Đề Thỏa Mãn Ràng Buộc (Constraint Satisfaction)
Các thuật toán này được thiết kế để tìm kiếm các lời giải thỏa mãn một tập hợp các ràng buộc đã cho.

**Thuật toán Backtracking Search (Tìm kiếm Quay lui):**
Đây là một thuật toán cơ bản cho các Vấn đề Thỏa mãn Ràng buộc (CSPs). Nó xây dựng lời giải một cách từ từ, gán giá trị cho từng biến một. Nếu tại một bước nào đó, một phép gán dẫn đến vi phạm ràng buộc, thuật toán sẽ "quay lui" và thử một giá trị khác.

**Thuật toán Forward Checking:**
Forward Checking là một cải tiến quan trọng của Backtracking. Khi một biến được gán một giá trị, thuật toán sẽ chủ động kiểm tra các biến chưa được gán và loại bỏ khỏi miền giá trị của chúng những giá trị không còn tương thích với phép gán hiện tại. Điều này giúp phát hiện các mâu thuẫn sớm hơn và cắt tỉa không gian tìm kiếm.

**Thuật toán AC-3 (Arc Consistency Algorithm 3):**
AC-3 là một thuật toán được sử dụng để đạt được "tính nhất quán cung" (arc consistency) cho các ràng buộc. Một cung (X, Y) là nhất quán nếu với mọi giá trị trong miền của X, tồn tại một giá trị trong miền của Y sao cho ràng buộc giữa X và Y được thỏa mãn. AC-3 có thể được sử dụng như một bước tiền xử lý hoặc tích hợp vào thuật toán tìm kiếm.

### Nhóm Thuật Toán Học Tăng Cường (Reinforcement Learning)
Trong học tăng cường, một agent học cách tương tác với một môi trường để tối đa hóa một tín hiệu phần thưởng tích lũy theo thời gian.

**Thuật toán Q-Learning:**
Q-Learning là một thuật toán học tăng cường không cần mô hình (model-free), hoạt động theo kiểu off-policy. Agent học một hàm giá trị-hành động, ký hiệu là Q(s, a), để ước tính "chất lượng" hay lợi ích kỳ vọng của việc thực hiện hành động 'a' khi đang ở trạng thái 's'. Mục tiêu là học được hàm Q tối ưu, từ đó suy ra chính sách tối ưu. Q-Learning sử dụng phương trình cập nhật Bellman để điều chỉnh các giá trị Q dựa trên kinh nghiệm thu được. Trong dự án này, Q-Learning được tăng cường với các kỹ thuật như experience replay (học lại từ các kinh nghiệm đã qua), cơ chế giảm dần cho các tham số khám phá (epsilon) và tốc độ học (alpha), cùng với một chiến lược sử dụng "bộ nhớ ngắn hạn" để tránh các vòng lặp không cần thiết và khuyến khích khám phá hiệu quả hơn.

## Hướng Dẫn Vận Hành Chi Tiết

### Khởi Động và Giao Diện Chính
Sau khi thực thi tệp `VuToanThang_23110329.py`, cửa sổ chính của **8-Puzzle Solver Pro** sẽ hiển thị. Giao diện được chia thành các khu vực chức năng chính:
-   **Khu vực trung tâm:** Nơi hiển thị trực quan bàn cờ 8-Puzzle.
-   **Panel bên trái (Algorithms Dock):** Danh sách các thuật toán, được phân nhóm một cách khoa học. Người dùng có thể nhấp vào tên thuật toán để lựa chọn.
-   **Panel bên phải (Controls Dock):** Tập hợp các nút điều khiển chức năng chính của ứng dụng, bao gồm "Đặt lại", "Ngẫu nhiên", "Giải", "Xuất", "Tự động", và "Đánh giá".
-   **Panel dưới cùng (Log Dock):** Cửa sổ nhật ký, hiển thị các thông báo hệ thống và ghi lại các hành động quan trọng của người dùng.
-   **Thanh trạng thái:** Nằm ở cuối cửa sổ, cung cấp thông tin nhanh về thuật toán hiện đang được chọn và tiến trình của lời giải (nếu có).

### Lựa Chọn và Cấu Hình Thuật Toán
Việc lựa chọn thuật toán được thực hiện bằng cách nhấp vào tên của thuật toán mong muốn trong danh sách ở panel bên trái. Tên của thuật toán được chọn sẽ ngay lập tức được cập nhật và hiển thị trên thanh trạng thái.

Một điểm đặc biệt là khi người dùng chọn thuật toán "Q-Learning" hoặc bất kỳ thuật toán nào thuộc nhóm "Tìm kiếm trong môi trường không xác định" (bao gồm AND/OR Graph Search, Fixed Partially Observable Search, và Belief State Search), một cửa sổ giao diện chuyên dụng sẽ tự động mở ra, cung cấp các công cụ và tùy chọn cấu hình riêng cho từng nhóm thuật toán này.

### Tương Tác Với Bàn Cờ 8-Puzzle
Người dùng có thể tương tác trực tiếp với bàn cờ 8-Puzzle thông qua các hành động sau:
-   **Di chuyển ô số:** Nhấp chuột vào một ô số nằm kề ô trống (ô số 0). Nếu hợp lệ, ô số đó sẽ di chuyển vào vị trí của ô trống.
-   **Nút "Đặt lại":** Khôi phục bàn cờ về trạng thái đích mặc định là `[1,2,3,4,5,6,7,0,8]`.
-   **Nút "Ngẫu nhiên":** Hệ thống sẽ tự động tạo ra một trạng thái bàn cờ mới, hoàn toàn ngẫu nhiên nhưng vẫn đảm bảo rằng trạng thái đó có thể giải được.
-   **Nút "Nhập liệu":** (Tính năng này đang được phát triển) Mục tiêu là cho phép người dùng tự định nghĩa một trạng thái bàn cờ cụ thể để bắt đầu.

### Thực Hiện Giải Puzzle
Quy trình giải một bài toán 8-Puzzle như sau:
1.  **Lựa chọn thuật toán:** Chọn một thuật toán phù hợp từ danh sách ở panel bên trái.
2.  **Thiết lập trạng thái ban đầu:** Sử dụng trạng thái mặc định, tạo một trạng thái ngẫu nhiên bằng nút "Ngẫu nhiên", hoặc tự di chuyển các ô để tạo trạng thái mong muốn.
3.  **Kích hoạt giải:** Nhấp vào nút **"Giải"** ở panel bên phải.
4.  Ứng dụng sẽ bắt đầu thực thi thuật toán đã chọn để tìm kiếm lời giải. Thời gian cần thiết cho quá trình này có thể thay đổi tùy thuộc vào độ phức tạp của trạng thái puzzle hiện tại và hiệu suất của thuật toán được sử dụng.
5.  **Hiển thị và tương tác với lời giải:**
    -   Nếu một lời giải được tìm thấy, chuỗi các bước di chuyển sẽ được lưu trữ.
    -   Các nút **"Previous"** và **"Next"**, nằm ngay phía dưới bàn cờ trong khu vực trung tâm, cho phép người dùng duyệt qua từng bước của lời giải một cách thủ công.
    -   Nút **"Tự động"** trong Controls Dock sẽ kích hoạt chế độ tự động phát lại toàn bộ các bước giải, kèm theo hiệu ứng hoạt họa di chuyển ô số.
6.  Thông tin về tổng số bước trong lời giải và bước hiện tại đang hiển thị sẽ được cập nhật trên thanh trạng thái.

### Vận Hành Chuyên Sâu Với Q-Learning
Khi thuật toán "Q-Learning" được chọn, một cửa sổ chuyên biệt có tiêu đề "Q-Learning - Học tăng cường" sẽ được mở, cung cấp một môi trường làm việc đầy đủ cho thuật toán này:
-   **Bảng Puzzle (trong cửa sổ QL):** Hiển thị trạng thái hiện tại của bàn cờ mà agent Q-Learning đang tương tác.
-   **Thông tin Trạng thái (trong cửa sổ QL):**
    -   Trực quan hóa các giá trị Q (Q-values) ước tính cho các hành động khả dĩ (Lên, Xuống, Trái, Phải) từ trạng thái hiện tại của bàn cờ QL.
    -   Sử dụng màu sắc và độ đậm để làm nổi bật hành động được coi là tốt nhất dựa trên các Q-values hiện tại.
-   **Tab "Thông tin huấn luyện":**
    -   Cung cấp thông tin cập nhật về tiến trình huấn luyện của agent, bao gồm số lượng "episodes" (lượt chơi thử) đã hoàn thành.
    -   Hiển thị các tham số hiện hành của agent, như Alpha (tốc độ học), Gamma (hệ số chiết khấu phần thưởng tương lai), Epsilon (tỷ lệ khám phá), và tổng số cặp (trạng thái, hành động) có giá trị Q đã được học.
    -   Trình bày một bản tóm tắt các kết quả đạt được sau mỗi phiên huấn luyện.
-   **Tab "Cấu hình":**
    -   Cho phép người dùng tùy chỉnh các siêu tham số quan trọng của agent Q-Learning, bao gồm: số lượng Episodes huấn luyện, giá trị Alpha, Gamma, và Epsilon ban đầu.
    -   **Nút "Bắt đầu huấn luyện":** Kích hoạt quá trình huấn luyện agent Q-Learning với các tham số đã được thiết lập. Quá trình này có thể yêu cầu một khoảng thời gian đáng kể.
    -   **Nút "Giải với Q-Learning":** Sau khi agent đã được huấn luyện, nút này cho phép sử dụng agent đó để tìm lời giải cho trạng thái puzzle hiện tại trên bàn cờ QL. Nếu tìm thấy, lời giải sẽ được hiển thị và có thể được tự động trình diễn trên cả bàn cờ trong cửa sổ QL và bàn cờ chính của ứng dụng.
-   **Cửa sổ Nhật ký (trong cửa sổ QL):** Ghi lại chi tiết các hoạt động và sự kiện liên quan đến quá trình huấn luyện và giải puzzle bằng Q-Learning.

### Khám Phá Môi Trường Không Xác Định
Khi người dùng lựa chọn một trong các thuật toán được thiết kế cho môi trường phức tạp, cụ thể là "AND/OR Graph Search", "Fixed Partially Observable Search", hoặc "Belief State Search", một cửa sổ chuyên dụng có tiêu đề "Tìm kiếm trong Môi Trường Không xác định" sẽ được kích hoạt. Cửa sổ này cung cấp các công cụ để mô phỏng và giải quyết bài toán 8-Puzzle trong điều kiện thông tin không hoàn hảo:
-   **Bảng Puzzle Thực tế:** Hiển thị trạng thái thực sự, đầy đủ của bàn cờ trong môi trường mô phỏng.
-   **Trạng thái Quan sát Được:** Trình bày những thông tin mà agent có thể "nhìn thấy" được, dựa trên cấu hình các vị trí có thể quan sát. Các ô không nằm trong tầm quan sát sẽ được hiển thị bằng ký hiệu "?".
-   **Thông tin Trạng thái Niềm Tin (Belief States):** Cung cấp thông tin về tập hợp các trạng thái mà agent cho là có thể xảy ra, dựa trên những gì nó quan sát được. Hiển thị số lượng trạng thái niềm tin hiện tại và trạng thái được đánh giá là có khả năng cao nhất.
-   **Cấu hình Môi trường:**
    -   Cho phép người dùng định nghĩa các vị trí (ô) trên bàn cờ mà agent có thể quan sát được (ví dụ, nhập chuỗi "0, 1, 2, 3, 4" để chỉ định 5 ô đầu tiên).
    -   Điều chỉnh xác suất để một hành động của agent dẫn đến một kết quả ngẫu nhiên, không như dự kiến (ví dụ, hành động "Lên" có thể khiến ô trống di chuyển sang "Trái").
    -   Thiết lập xác suất để môi trường tự động thay đổi trạng thái một cách ngẫu nhiên (ví dụ, một ô số tự di chuyển).
    -   Nút "Cập nhật Cấu hình" dùng để áp dụng các thay đổi này vào môi trường mô phỏng.
-   **Lựa chọn Thuật toán (trong cửa sổ này):** Cho phép chuyển đổi giữa ba thuật toán chuyên dụng: AND/OR Graph Search, Fixed Partially Observable Search, và Belief State Search.
-   **Các nút điều khiển chính:**
    -   **"Giải":** Thực thi thuật toán đã chọn để tìm kiếm một chuỗi hành động giải quyết puzzle trong môi trường không xác định hiện tại.
    -   **"Đặt lại":** Tạo ra một trạng thái puzzle ngẫu nhiên mới cho môi trường.
    -   **"Bước tiếp theo":** Nếu một lời giải (dưới dạng chuỗi hành động) đã được tìm thấy, nút này sẽ thực hiện hành động kế tiếp trong chuỗi đó trên môi trường.
    -   **"Tự động Chạy":** Tự động thực hiện tuần tự các hành động trong lời giải đã tìm được.
-   **Thông tin Giải pháp:** Hiển thị chuỗi hành động của lời giải (nếu tìm thấy).
-   **Cửa sổ Nhật ký (trong cửa sổ Non-Deterministic):** Ghi lại tất cả các hoạt động và sự kiện xảy ra trong quá trình tương tác với môi trường không xác định.

### Phân Tích và Đánh Giá Hiệu Suất Thuật Toán
1.  Để bắt đầu quá trình đánh giá, người dùng nhấp vào nút **"Đánh giá"** từ Controls Dock trên giao diện chính. Thao tác này sẽ mở cửa sổ "Đánh giá Thuật toán 8 Puzzle".
2.  **Thiết lập tham số đánh giá:**
    -   **Số lượng Puzzle:** Xác định số lượng các trạng thái puzzle ngẫu nhiên sẽ được tạo ra để phục vụ cho việc đánh giá (giá trị mặc định thường là 10).
    -   **Nhóm Thuật toán:** Lựa chọn nhóm thuật toán cụ thể mà người dùng muốn tiến hành đánh giá (ví dụ: "Tất cả các thuật toán", "Chỉ nhóm Tìm kiếm có thông tin", v.v.).
3.  Sau khi cấu hình, nhấp vào nút **"Chạy Đánh giá"**.
4.  Quá trình đánh giá các thuật toán sẽ được thực thi ngầm (sử dụng QThread của PyQt5), đảm bảo giao diện người dùng vẫn phản hồi và không bị "đóng băng".
5.  **Theo dõi tiến độ:**
    -   Một thanh tiến trình sẽ trực quan hóa phần trăm công việc đã hoàn thành.
    -   Một nhãn trạng thái sẽ liên tục cập nhật thông tin về thuật toán hiện đang được kiểm thử và puzzle cụ thể đang được xử lý.
6.  **Trình bày kết quả phân tích:** Khi quá trình đánh giá hoàn tất, kết quả sẽ được tổng hợp và trình bày một cách chi tiết thông qua các tab giao diện:
    -   **Tab "Bảng So sánh":** Hiển thị một bảng dữ liệu chi tiết, liệt kê các số liệu hiệu suất cho mỗi thuật toán, bao gồm: Tên thuật toán, Thời gian thực thi trung bình (tính bằng giây), Độ dài trung bình của đường đi tìm được, Tỷ lệ giải thành công (tính bằng phần trăm), và Ước tính số lượng nút đã được mở rộng trong quá trình tìm kiếm.
    -   **Tab "Thời gian Thực thi":** Một biểu đồ cột trực quan hóa sự khác biệt về thời gian chạy trung bình giữa các thuật toán.
    -   **Tab "Độ dài Đường đi":** Biểu đồ cột so sánh độ dài trung bình của các lời giải được tìm thấy bởi mỗi thuật toán.
    -   **Tab "Tỷ lệ Thành công":** Biểu đồ cột minh họa tỷ lệ phần trăm các bài toán puzzle mà mỗi thuật toán đã giải quyết thành công.
    -   **Tab "So sánh Theo Nhóm":** Sử dụng một giao diện dạng thẻ (card-based) hiện đại và trực quan, tab này cung cấp một cái nhìn sâu sắc về hiệu suất của từng thuật toán trong mỗi nhóm phân loại. Mỗi thẻ thuật toán sẽ hiển thị các biểu đồ nhỏ và thông tin tóm tắt về hiệu suất của nó.
7.  Để lưu trữ một bản sao của báo cáo đánh giá, người dùng có thể nhấp vào nút **"Xuất PDF"**.

### Khai Thác và Lưu Trữ Kết Quả
Ứng dụng cung cấp hai cơ chế chính để lưu trữ và khai thác kết quả:
-   **Xuất Lời Giải Chi Tiết:** Sau khi một puzzle được giải thành công bằng một thuật toán bất kỳ (ngoại trừ Q-Learning và các thuật toán cho môi trường không xác định, do chúng có cơ chế hiển thị và tương tác riêng trong các cửa sổ chuyên dụng), người dùng có thể nhấp vào nút **"Xuất"** trong Controls Dock. Thao tác này sẽ lưu trữ chi tiết các bước của lời giải vào một tệp văn bản có tên `solution.txt`.
-   **Xuất Báo Cáo Đánh Giá PDF:** Bên trong cửa sổ "Đánh giá Thuật toán", sau khi một phiên đánh giá đã được thực hiện và kết quả đã được hiển thị, nút **"Xuất PDF"** sẽ khả dụng. Khi nhấp vào đây, hệ thống sẽ yêu cầu người dùng chọn vị trí và tên tệp để lưu báo cáo. Báo cáo PDF được tạo ra sẽ chứa đựng đầy đủ thông tin, bao gồm bảng so sánh chi tiết và tất cả các biểu đồ trực quan đã được tạo ra trong quá trình phân tích.

## Phân Tích Hiệu Suất và Kết Quả Đánh Giá

Phần này dành để trình bày các kết quả phân tích hiệu suất chi tiết của các thuật toán đã được triển khai. Các biểu đồ dưới đây được trích xuất từ báo cáo PDF do chính ứng dụng tạo ra, minh họa một cách trực quan sự khác biệt về hiệu quả giữa các phương pháp tiếp cận.

## Hướng Dẫn Cài Đặt và Triển Khai

### Yêu Cầu Hệ Thống và Thư Viện
Để vận hành ứng dụng 8-Puzzle Solver Pro, môi trường của bạn cần đáp ứng các yêu cầu sau:
-   **Python:** Phiên bản 3.x (khuyến nghị 3.7 trở lên).
-   **Thư viện PyQt5:** Dành cho việc xây dựng giao diện đồ họa người dùng.
-   **Thư viện Matplotlib:** Được sử dụng để vẽ các biểu đồ trong chức năng đánh giá thuật toán.
-   **Thư viện ReportLab:** Cần thiết cho việc tạo và xuất các báo cáo đánh giá dưới định dạng PDF.
-   **Thư viện NumPy:** Hỗ trợ các thao tác tính toán số học hiệu quả, đặc biệt quan trọng cho một số thuật toán và xử lý dữ liệu.
-   **Thư viện Pandas:** Chủ yếu được sử dụng trong mô-đun đánh giá để xử lý và cấu trúc dữ liệu kết quả, và có thể được tận dụng cho các tính năng phân tích dữ liệu nâng cao trong tương lai.

### Quy Trình Cài Đặt
1.  **Tải mã nguồn:** Nếu dự án được lưu trữ trên một kho chứa mã nguồn (ví dụ: GitHub), hãy clone kho chứa đó về máy của bạn. Hoặc, tải trực tiếp tệp mã nguồn `VuToanThang_23110329.py`.
2.  **Cài đặt các thư viện phụ thuộc:** Mở terminal hoặc command prompt và sử dụng pip (trình quản lý gói của Python) để cài đặt tất cả các thư viện cần thiết. Thực thi lệnh sau:
    ```bash
    pip install PyQt5 matplotlib reportlab numpy pandas
    ```
3.  **Lưu ý về Font chữ cho xuất PDF (đối với người dùng Windows):** Chức năng xuất báo cáo PDF có thể yêu cầu font chữ `Arial` (tệp `arial.ttf`). Đảm bảo rằng font này có sẵn trên thư mục hệ thống `C:\Windows\Fonts\`. Nếu bạn sử dụng một hệ điều hành khác hoặc gặp sự cố liên quan đến font khi xuất PDF, bạn có thể cần phải điều chỉnh đường dẫn đến tệp font chữ trong mã nguồn của ứng dụng (cụ thể là trong lớp `EvaluationPage`, phương thức `export_to_pdf`) để trỏ đến một tệp font tương thích có sẵn trên hệ thống của bạn.

### Khởi Chạy Ứng Dụng
Sau khi hoàn tất các bước cài đặt, bạn có thể khởi chạy ứng dụng bằng cách thực thi tệp Python chính từ terminal hoặc command prompt:
```bash
python VuToanThang_23110329.py
```
Cửa sổ chính của ứng dụng **8-Puzzle Solver Pro** sẽ được hiển thị, sẵn sàng cho bạn khám phá và sử dụng.
